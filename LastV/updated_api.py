import os
import uvicorn
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Input, GlobalAveragePooling1D, LayerNormalization, Lambda
from tensorflow.keras import Model
from ultralytics import YOLO

print("TensorFlow version:", tf.__version__)

# ========= CUSTOM LAYERS =========
class MultiHeadAttention(Layer):
    def __init__(self, num_heads=4, key_dim=32, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.d_model = num_heads * key_dim
        self.query_dense = Dense(self.d_model)
        self.key_dense = Dense(self.d_model)
        self.value_dense = Dense(self.d_model)
        self.combine_heads = Dense(self.d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.key_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, query, value, key):
        batch_size = tf.shape(query)[0]
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, value)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))
        return self.combine_heads(concat_attention)

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({"num_heads": self.num_heads, "key_dim": self.key_dim})
        return config


class VisionTemporalTransformer(Layer):
    def __init__(self, patch_size=8, d_model=128, num_heads=4, spatial_layers=1, temporal_layers=1, **kwargs):
        super(VisionTemporalTransformer, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.spatial_layers = spatial_layers
        self.temporal_layers = temporal_layers
        self.dense_projection = Dense(d_model)
        self.pos_emb = None
        self.spatial_mhas = [MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads) for _ in range(spatial_layers)]
        self.spatial_norm1 = [LayerNormalization() for _ in range(spatial_layers)]
        self.spatial_ffn = [tf.keras.Sequential([Dense(d_model*4, activation="relu"), Dense(d_model)]) for _ in range(spatial_layers)]
        self.spatial_norm2 = [LayerNormalization() for _ in range(spatial_layers)]
        self.temporal_mhas = [MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads) for _ in range(temporal_layers)]
        self.temporal_norm1 = [LayerNormalization() for _ in range(temporal_layers)]
        self.temporal_ffn = [tf.keras.Sequential([Dense(d_model*4, activation="relu"), Dense(d_model)]) for _ in range(temporal_layers)]
        self.temporal_norm2 = [LayerNormalization() for _ in range(temporal_layers)]

    def build(self, input_shape):
        H, W = input_shape[2], input_shape[3]
        ph, pw = H // self.patch_size, W // self.patch_size
        num_patches = ph * pw
        self.pos_emb = self.add_weight(shape=(1, num_patches, self.d_model), initializer="random_normal", trainable=True, name="pos_emb")
        super(VisionTemporalTransformer, self).build(input_shape)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch, frames, H, W = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        reshaped = tf.reshape(inputs, (-1, H, W, tf.shape(inputs)[-1]))
        patches = tf.image.extract_patches(reshaped, sizes=[1, self.patch_size, self.patch_size, 1],
                                          strides=[1, self.patch_size, self.patch_size, 1], rates=[1,1,1,1], padding="VALID")
        patches = tf.reshape(patches, (-1, tf.shape(patches)[1]*tf.shape(patches)[2], tf.shape(patches)[-1]))
        x = self.dense_projection(patches) + self.pos_emb

        # Spatial transformer
        for mha, n1, ffn, n2 in zip(self.spatial_mhas, self.spatial_norm1, self.spatial_ffn, self.spatial_norm2):
            attn = mha(x, x, x)
            x = n1(x + attn)
            ff = ffn(x)
            x = n2(x + ff)

        # Temporal pooling
        x = tf.reshape(x, (batch, frames, -1, self.d_model))
        x = tf.reduce_mean(x, axis=2)

        # Temporal transformer
        for mha, n1, ffn, n2 in zip(self.temporal_mhas, self.temporal_norm1, self.temporal_ffn, self.temporal_norm2):
            attn = mha(x, x, x)
            x = n1(x + attn)
            ff = ffn(x)
            x = n2(x + ff)

        return GlobalAveragePooling1D()(x)

    def get_config(self):
        config = super(VisionTemporalTransformer, self).get_config()
        config.update({
            "patch_size": self.patch_size, "d_model": self.d_model, "num_heads": self.num_heads,
            "spatial_layers": self.spatial_layers, "temporal_layers": self.temporal_layers
        })
        return config


# ========= YOLO FACE VIDEO PROCESSOR (FROM APP.PY) =========
class YOLOFaceVideoProcessor:
    def __init__(self, frame_count=8, dim=(64,144), yolo_model_path='yolov8n.pt',
                 candidate_frames=24, shrink_factor=0.20, apply_sharpen=True):
        self.frame_count = frame_count
        self.dim = dim
        self.candidate_frames = candidate_frames
        self.shrink_factor = shrink_factor
        self.apply_sharpen = apply_sharpen
        # Use relative path - yolo model in same directory
        if not os.path.isabs(yolo_model_path):
            yolo_model_path = os.path.join(os.path.dirname(__file__), yolo_model_path)
        self.detector = YOLO(yolo_model_path)
        self.sharp_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def _read_all_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        cap.release()
        return all_frames

    def _sample_candidates(self, all_frames):
        n = len(all_frames)
        if n == 0:
            return []
        if n <= self.candidate_frames:
            return list(range(n))
        indices = np.linspace(0, n-1, self.candidate_frames, dtype=int).tolist()
        return indices

    def _optical_flow_scores(self, all_frames, candidate_idx):
        scores = np.zeros(len(candidate_idx))
        gray_frames = [cv2.cvtColor(all_frames[i], cv2.COLOR_BGR2GRAY) for i in candidate_idx]
        for i in range(1, len(gray_frames)):
            prev = gray_frames[i-1]
            cur = gray_frames[i]
            flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5,3,15,3,5,1.2,0)
            scores[i] = np.mean(np.abs(flow))

        diff_scores = []
        for idx in candidate_idx:
            if idx == 0:
                diff_scores.append(0.0)
            else:
                prev = cv2.cvtColor(all_frames[idx-1], cv2.COLOR_BGR2GRAY)
                cur = cv2.cvtColor(all_frames[idx], cv2.COLOR_BGR2GRAY)
                diff_scores.append(float(np.mean(np.abs(cur.astype(np.float32)-prev.astype(np.float32)))))
        scores += np.array(diff_scores)
        return scores

    def detect_and_crop_face(self, frame):
        results = self.detector(frame)[0]
        if len(results.boxes) == 0:
            return cv2.resize(frame, (self.dim[1], self.dim[0]), interpolation=cv2.INTER_CUBIC)

        box = results.boxes.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        w = x2 - x1
        h = y2 - y1
        shrink_w = int(w * self.shrink_factor)
        shrink_h = int(h * self.shrink_factor)
        x1n = max(0, x1 + shrink_w)
        y1n = max(0, y1 + shrink_h)
        x2n = min(frame.shape[1], x2 - shrink_w)
        y2n = min(frame.shape[0], y2 - shrink_h)

        if x2n <= x1n or y2n <= y1n:
            x1n, y1n, x2n, y2n = x1, y1, x2, y2

        face_crop = frame[y1n:y2n, x1n:x2n]

        if self.apply_sharpen:
            face_crop = cv2.filter2D(face_crop, -1, self.sharp_kernel)

        try:
            ycrcb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            y = self.clahe.apply(cv2.convertScaleAbs(y))
            face_crop = cv2.cvtColor(cv2.merge((y, cr, cb)), cv2.COLOR_YCrCb2BGR)
        except:
            pass

        return cv2.resize(face_crop, (self.dim[1], self.dim[0]), interpolation=cv2.INTER_CUBIC)

    def select_transition_frames(self, video_path):
        all_frames = self._read_all_frames(video_path)
        if len(all_frames) == 0:
            blank = np.zeros((self.dim[0], self.dim[1], 3), dtype=np.float32)
            return np.array([blank.copy() for _ in range(self.frame_count)])

        candidate_idx = self._sample_candidates(all_frames)
        scores = self._optical_flow_scores(all_frames, candidate_idx)
        top_k = min(self.frame_count, len(candidate_idx))
        chosen_positions = np.argsort(scores)[-top_k:]
        chosen_idx = sorted([candidate_idx[i] for i in chosen_positions])

        if len(chosen_idx) < self.frame_count:
            last = chosen_idx[-1] if chosen_idx else 0
            for i in range(self.frame_count - len(chosen_idx)):
                chosen_idx.append(min(len(all_frames)-1, last + i + 1))

        processed = []
        for idx in chosen_idx[:self.frame_count]:
            f = all_frames[idx]
            processed.append(self.detect_and_crop_face(f).astype(np.float32))

        while len(processed) < self.frame_count:
            processed.append(np.zeros((self.dim[0], self.dim[1], 3), dtype=np.float32))

        frames = np.stack(processed, axis=0)
        return np.clip(frames / 255.0, 0.0, 1.0).astype(np.float32)

    def compute_residue(self, frames):
        residues = np.zeros((self.frame_count-1, self.dim[0], self.dim[1], 3), dtype=np.float32)
        for i in range(1, len(frames)):
            residues[i-1] = frames[i] - frames[i-1]
        return residues

    def load_video(self, path):
        frames = self.select_transition_frames(path)
        residues = self.compute_residue(frames)
        return frames, residues


# ========= BUILD MODEL =========
def build_model():
    frame_input = Input(shape=(8,64,144,3))
    residue_input = Input(shape=(7,64,144,3))

    vt = VisionTemporalTransformer(
        patch_size=8,
        d_model=128,
        num_heads=4,
        spatial_layers=1,
        temporal_layers=1
    )

    frame_feat = vt(frame_input)
    residue_feat = vt(residue_input)

    expand1 = Lambda(lambda x: tf.expand_dims(x,axis=1))(frame_feat)
    q = expand1
    k = Lambda(lambda x: tf.expand_dims(x,axis=1))(residue_feat)
    v = k

    mha = MultiHeadAttention(num_heads=4, key_dim=32)
    attn_out = mha(q, value=v, key=k)
    attn_out = Lambda(lambda x: tf.squeeze(x,axis=1))(attn_out)

    fusion = Lambda(lambda t: tf.concat(t,axis=1))([frame_feat, residue_feat, attn_out])

    x = Dense(512, activation='relu')(fusion)
    x = Dense(256, activation='relu')(x)
    class_output = Dense(2, activation='softmax')(x)
    features_output = Dense(128, activation=None)(x)

    return Model(inputs=[frame_input, residue_input], outputs=[class_output, features_output])


# ========= FastAPI =========
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Use relative paths - files in same directory
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "best_lipinc_model.h5")
YOLO_PATH = "yolov8n.pt"  # Will be resolved relative to this file

print(f"🔄 Building and loading model from: {MODEL_PATH}")

# Build model and load weights
model = build_model()
model.load_weights(MODEL_PATH)
print("✅ Model rebuilt and weights loaded successfully!")
print(f"Model inputs: {[x.shape for x in model.inputs]}")

# Initialize video processor with YOLO
video_processor = YOLOFaceVideoProcessor(
    frame_count=8,
    dim=(64,144),
    yolo_model_path=YOLO_PATH,
    candidate_frames=24,
    shrink_factor=0.20,
    apply_sharpen=True
)


class PredictionResponse(BaseModel):
    result: str
    confidence: float
    prob_real: float
    prob_fake: float
    filename: str


@app.get("/")
def root():
    return {"message": "LIPINC Deepfake Detection API with YOLO ✅"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_video(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    try:
        # Save uploaded video
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Process video with YOLO face detection
        frames, residues = video_processor.load_video(temp_path)

        # Prepare batches
        frames_batch = np.expand_dims(frames, 0)
        residues_batch = np.expand_dims(residues, 0)

        # Predict
        preds = model.predict([frames_batch, residues_batch], verbose=0)
        class_probs = preds[0][0]
        prob_real, prob_fake = class_probs

        result = "Real" if prob_real >= prob_fake else "Fake"
        confidence = max(prob_real, prob_fake)

        return PredictionResponse(
            result=result,
            confidence=float(confidence),
            prob_real=float(prob_real),
            prob_fake=float(prob_fake),
            filename=file.filename
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
