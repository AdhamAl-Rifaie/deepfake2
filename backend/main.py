import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set environment variable BEFORE importing tensorflow/keras
import sys
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import numpy as np

# Import tensorflow and immediately enable unsafe deserialization
import tensorflow as tf
from tensorflow import keras
keras.config.enable_unsafe_deserialization()

from tensorflow.keras.layers import Dense, Layer, GlobalAveragePooling1D, LayerNormalization
import cv2
import tempfile
import json

app = FastAPI()

# ... rest of your code


# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== CUSTOM LAYERS ====================
# These MUST match your training notebook exactly

class MultiHeadAttention(Layer):
    """Custom Multi-Head Attention layer"""
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
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        
        return self.combine_heads(output)
    
    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
        })
        return config


class VisionTemporalTransformer(Layer):
    """Vision Temporal Transformer for video processing"""
    def __init__(self, patch_size=8, d_model=128, num_heads=4, 
                 spatial_layers=1, temporal_layers=1, **kwargs):
        super(VisionTemporalTransformer, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.spatial_layers = spatial_layers
        self.temporal_layers = temporal_layers
        
        self.dense_projection = Dense(d_model)
        self.pos_emb = None
        
        # Spatial transformer components
        self.spatial_mhas = [
            MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)
            for _ in range(spatial_layers)
        ]
        self.spatial_norm1 = [LayerNormalization() for _ in range(spatial_layers)]
        self.spatial_ffn = [
            tf.keras.Sequential([
                Dense(d_model * 4, activation='relu'),
                Dense(d_model)
            ]) for _ in range(spatial_layers)
        ]
        self.spatial_norm2 = [LayerNormalization() for _ in range(spatial_layers)]
        
        # Temporal transformer components
        self.temporal_mhas = [
            MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)
            for _ in range(temporal_layers)
        ]
        self.temporal_norm1 = [LayerNormalization() for _ in range(temporal_layers)]
        self.temporal_ffn = [
            tf.keras.Sequential([
                Dense(d_model * 4, activation='relu'),
                Dense(d_model)
            ]) for _ in range(temporal_layers)
        ]
        self.temporal_norm2 = [LayerNormalization() for _ in range(temporal_layers)]
    
    def build(self, input_shape):
        H, W = input_shape[2], input_shape[3]
        ph, pw = H // self.patch_size, W // self.patch_size
        num_patches = ph * pw
        
        self.pos_emb = self.add_weight(
            shape=(1, num_patches, self.d_model),
            initializer='random_normal',
            trainable=True,
            name='pos_emb'
        )
        super(VisionTemporalTransformer, self).build(input_shape)
    
    def call(self, inputs):
        input_shape = inputs.get_shape()
        shape = tf.shape(inputs)
        batch, frames, H, W = shape[0], shape[1], shape[2], shape[3]
        C_static = input_shape[-1]
        C = C_static if C_static is not None else shape[4]
        
        reshaped = tf.reshape(inputs, [-1, H, W, C])
        
        patches = tf.image.extract_patches(
            images=reshaped,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        
        if C_static is not None:
            patch_dim_static = self.patch_size * self.patch_size * C_static
        else:
            patch_dim_static = None
        
        patch_dim_dynamic = tf.shape(patches)[-1]
        final_patch_dim = patch_dim_static if patch_dim_static is not None else patch_dim_dynamic
        
        patches = tf.reshape(patches, [-1, tf.shape(patches)[1] * tf.shape(patches)[2], final_patch_dim])
        
        if patch_dim_static is not None:
            patches.set_shape([None, None, patch_dim_static])
        
        x = self.dense_projection(patches) + self.pos_emb
        
        # Spatial transformer
        for i in range(self.spatial_layers):
            attn = self.spatial_mhas[i](x, x, x)
            x = x + attn
            x = self.spatial_norm1[i](x)
            ff = self.spatial_ffn[i](x)
            x = x + ff
            x = self.spatial_norm2[i](x)
        
        # Temporal pooling
        x = tf.reshape(x, [batch, frames, -1, self.d_model])
        x = tf.reduce_mean(x, axis=2)
        x.set_shape([None, None, self.d_model])
        
        # Temporal transformer
        for i in range(self.temporal_layers):
            attn = self.temporal_mhas[i](x, x, x)
            x = x + attn
            x = self.temporal_norm1[i](x)
            ff = self.temporal_ffn[i](x)
            x = x + ff
            x = self.temporal_norm2[i](x)
        
        pooled = GlobalAveragePooling1D()(x)
        return pooled
    
    def get_config(self):
        config = super(VisionTemporalTransformer, self).get_config()
        config.update({
            "patch_size": self.patch_size,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "spatial_layers": self.spatial_layers,
            "temporal_layers": self.temporal_layers,
        })
        return config


# ==================== LOAD MODEL ====================
MODEL_PATH = "best_lipinc_model.h5"

custom_objects = {
    'MultiHeadAttention': MultiHeadAttention,
    'VisionTemporalTransformer': VisionTemporalTransformer
}

# Import h5py to bypass some Keras 3.x restrictions
import h5py

print("Loading model...")
try:
    # Force TensorFlow backend and load
    with h5py.File(MODEL_PATH, 'r') as f:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"Standard load failed: {e}")
    # Alternative: Load architecture and weights separately
    from tensorflow.keras.models import model_from_json
    print("Attempting alternative loading...")
    with h5py.File(MODEL_PATH, 'r') as f:
        model_config = f.attrs.get('model_config')
        if model_config is None:
            raise ValueError('No model config found')
        model_config = json.loads(model_config.decode('utf-8'))
        model = model_from_json(json.dumps(model_config), custom_objects=custom_objects)
        model.load_weights(MODEL_PATH)
    print("✓ Model loaded with alternative method!")

# Configuration
FRAME_COUNT = 8
DIM = (64, 144)
USE_TRANSITIONS = True



# ==================== VIDEO PROCESSING ====================
def select_transition_frames(video_path, frame_count=8, dim=(64, 144)):
    """Select frames at scene transitions/changes"""
    cap = cv2.VideoCapture(video_path)
    prev_gray = None
    transition_scores = []
    all_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            mean_diff = np.mean(diff)
            transition_scores.append(mean_diff)
        else:
            transition_scores.append(0)
            
        all_frames.append(frame)
        prev_gray = gray
    
    cap.release()
    
    if len(all_frames) >= frame_count:
        top_indices = np.argsort(transition_scores)[-frame_count:]
        top_indices = sorted(top_indices)
    else:
        top_indices = list(range(len(all_frames)))
    
    frames = []
    for idx in top_indices[:frame_count]:
        frame = all_frames[idx]
        frame = cv2.resize(frame, (dim[1], dim[0]))
        frames.append(frame)
    
    while len(frames) < frame_count:
        frames.append(np.zeros((dim[0], dim[1], 3)))
    
    return np.array(frames).astype(np.float32) / 255.0


def load_video_uniform(video_path, frame_count=8, dim=(64, 144)):
    """Load frames uniformly distributed across video"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    
    if total_frames >= frame_count:
        indices = np.linspace(0, total_frames - 1, frame_count, dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (dim[1], dim[0]))
                frames.append(frame)
    else:
        while len(frames) < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (dim[1], dim[0]))
            frames.append(frame)
    
    cap.release()
    
    while len(frames) < frame_count:
        frames.append(np.zeros((dim[0], dim[1], 3)))
    
    return np.array(frames).astype(np.float32) / 255.0


def load_video(video_path):
    """Main video loading function"""
    try:
        if USE_TRANSITIONS:
            return select_transition_frames(video_path, FRAME_COUNT, DIM)
        else:
            return load_video_uniform(video_path, FRAME_COUNT, DIM)
    except Exception as e:
        print(f"Warning: Error loading {video_path}, using uniform sampling. Error: {e}")
        return load_video_uniform(video_path, FRAME_COUNT, DIM)


def compute_residue(frames):
    """Compute residue frames (frame differences)"""
    residues = np.zeros((FRAME_COUNT - 1, DIM[0], DIM[1], 3), dtype=np.float32)
    
    if len(frames) > 1:
        for i in range(1, len(frames)):
            residues[i - 1] = frames[i] - frames[i - 1]
    
    return residues


# ==================== API ENDPOINTS ====================
@app.post("/predict")
async def predict_deepfake(file: UploadFile = File(...)):
    """Predict if uploaded video is Real or Fake"""
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=415, detail="Only video files are accepted")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        file_path = temp_file.name
        shutil.copyfileobj(file.file, temp_file)
    
    try:
        print(f"Processing video: {file.filename}")
        
        frames = load_video(file_path)
        residues = compute_residue(frames)
        
        frames_input = np.expand_dims(frames, axis=0)
        residues_input = np.expand_dims(residues, axis=0)
        
        predictions = model.predict([frames_input, residues_input], verbose=0)
        
        class_probs = predictions[0][0]
        prob_real = float(class_probs[0])
        prob_fake = float(class_probs[1])
        
        result = "Real" if prob_real > prob_fake else "Fake"
        confidence = max(prob_real, prob_fake)
        
        print(f"Result: {result} (confidence: {confidence:.2%})")
        
        return {
            "result": result,
            "confidence": confidence,
            "prob_real": prob_real,
            "prob_fake": prob_fake,
            "filename": file.filename
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.get("/")
def read_root():
    return {
        "status": "Deepfake Detection API Running",
        "model": "LIPINC (Vision Temporal Transformer)",
        "frame_extraction": "Transition Detection" if USE_TRANSITIONS else "Uniform Sampling"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "frame_count": FRAME_COUNT,
        "dimensions": DIM
    }
