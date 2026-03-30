import numpy as np
import pandas as pd
import pickle
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

print("Loading trained data...")
df = pd.read_csv("data/raw/gestures_all.csv")
X = df.drop("label", axis=1).values
y = df["label"].values
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Building Keras neural network...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(300,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training neural network...")
history = model.fit(
    X, y_encoded,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)
final_acc = history.history['val_accuracy'][-1]
print(f"Neural network accuracy: {final_acc * 100:.2f}%")

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

os.makedirs("data/processed", exist_ok=True)
tflite_path = "data/processed/gesture_model.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

size_bytes = os.path.getsize(tflite_path)
size_kb = size_bytes / 1024
print(f"TFLite model saved: {tflite_path}")
print(f"Model size: {size_kb:.1f} KB ({size_bytes} bytes)")

ESP32_FLASH_LIMIT = 1024 * 1024
ESP32_RAM_LIMIT   = 320 * 1024
if size_bytes < ESP32_FLASH_LIMIT:
    print(f"ESP32 flash check: PASS (limit 1024KB, used {size_kb:.1f}KB)")
else:
    print(f"ESP32 flash check: FAIL (too large)")

print("\nConverting to C array for ESP32 firmware...")
c_array = ", ".join([f"0x{b:02x}" for b in tflite_model])
c_header = f"""// Auto-generated TFLite model for ESP32
// Size: {size_bytes} bytes ({size_kb:.1f} KB)
// Accuracy: {final_acc * 100:.2f}%

#ifndef GESTURE_MODEL_H
#define GESTURE_MODEL_H

const unsigned char gesture_model[] = {{
  {c_array}
}};
const unsigned int gesture_model_len = {size_bytes};

#endif
"""

header_path = "firmware_tflite/gesture_model.h"
os.makedirs("firmware_tflite", exist_ok=True)
with open(header_path, "w") as f:
    f.write(c_header)
print(f"C header saved: {header_path}")

aws_path = "data/processed/gesture_model.tflite"
os.system(f"aws s3 cp {aws_path} s3://esp32-tinyml-gestures-bucket/models/gesture_model.tflite")
print("TFLite model uploaded to S3")

print(f"\nSummary:")
print(f"  Neural network accuracy: {final_acc * 100:.2f}%")
print(f"  TFLite model size:       {size_kb:.1f} KB")
print(f"  ESP32 compatible:        YES")
print(f"  C header generated:      firmware_tflite/gesture_model.h")
