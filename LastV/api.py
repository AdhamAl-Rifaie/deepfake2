import os
import uvicorn
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Input, GlobalAveragePooling1D, LayerNormalization, Lambda, Dropout
from tensorflow.keras import Model

print("TensorFlow version:", tf.__version__)

# ========= EXACT CUSTOM LAYERS FROM YOUR NOTEBOOK =========
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

# ========= REBUILD MODEL EXACTLY LIKE NOTEBOOK =========
def build_lipinc_model(frame_shape=(8, 64, 144, 3), residue_shape=(7, 64, 144, 3), d_model=128):
    frame_input = Input(shape=frame_shape, name="FrameInput")
    residue_input = Input(shape=residue_shape, name="ResidueInput")
    
    vt = VisionTemporalTransformer(patch_size=8, d_model=d_model, num_heads=4, spatial_layers=1, temporal_layers=1)
    frame_feat = vt(frame_input)
    residue_feat = vt(residue_input)
    
    # Cross-attention
    q = Lambda(lambda x: tf.expand_dims(x, axis=1))(frame_feat)
    k = Lambda(lambda x: tf.expand_dims(x, axis=1))(residue_feat)
    v = k
    mha = MultiHeadAttention(num_heads=4, key_dim=d_model//4)
    attn_out = mha(q, v, k)
    attn_out = Lambda(lambda x: tf.squeeze(x, axis=1))(attn_out)
    
    # Feature fusion
    fusion = Lambda(lambda t: tf.concat(t, axis=1))([frame_feat, residue_feat, attn_out])
    
    # Classification head
    x = Dense(512, activation='relu')(fusion)
    x = Dense(256, activation='relu')(x)
    class_output = Dense(2, activation='softmax', name='classoutput')(x)
    features_output = Dense(d_model, activation=None, name='featuresoutput')(x)
    
    model = Model(inputs=[frame_input, residue_input], outputs=[class_output, features_output], name="LIPINCModel")
    return model

# ========= VideoProcessor =========
class VideoProcessor:
    def __init__(self, frame_count=8, dim=(64, 144)):
        self.frame_count = frame_count
        self.dim = dim

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        
        if total_frames >= self.frame_count:
            indices = np.linspace(0, total_frames-1, self.frame_count, dtype=int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (self.dim[1], self.dim[0]))
                    frames.append(frame)
        else:
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.resize(frame, (self.dim[1], self.dim[0]))
                frames.append(frame)
        
        cap.release()
        while len(frames) < self.frame_count:
            frames.append(np.zeros((self.dim[0], self.dim[1], 3), dtype=np.float32))
        return np.array(frames).astype(np.float32) / 255.0

    def compute_residue(self, frames):
        residues = np.zeros((self.frame_count-1, *self.dim, 3), dtype=np.float32)
        for i in range(1, len(frames)):
            residues[i-1] = frames[i] - frames[i-1]
        return residues

# ========= FastAPI =========
app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_lipinc_model.h5")
print(f"🔄 Building and loading model from: {MODEL_PATH}")

# Build fresh model and load weights
model = build_lipinc_model()
model.load_weights(MODEL_PATH)
print("✅ Model rebuilt and weights loaded successfully!")
print(f"Model inputs: {[x.shape for x in model.inputs]}")

video_processor = VideoProcessor()

class PredictionResponse(BaseModel):
    result: str
    confidence: float
    prob_real: float
    prob_fake: float
    filename: str

@app.get("/")
def root():
    return {"message": "LIPINC Deepfake Detection API ✅"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_video(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        frames = video_processor.load_video(temp_path)
        residues = video_processor.compute_residue(frames)
        
        frames_batch = np.expand_dims(frames, 0)
        residues_batch = np.expand_dims(residues, 0)
        
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
