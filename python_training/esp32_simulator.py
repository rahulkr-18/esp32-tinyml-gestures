import requests
import numpy as np
import json
import time
import random
from datetime import datetime

API_URL = "https://h6l0h9opq5.execute-api.us-east-1.amazonaws.com/prod/predict"
DEVICE_ID = "esp32-gesture-device-sim"
TIMESTEPS = 50

def generate_wave():
    t = np.linspace(0, 2 * np.pi, TIMESTEPS)
    ax = np.sin(2 * t) + np.random.normal(0, 0.1, TIMESTEPS)
    ay = np.random.normal(0, 0.1, TIMESTEPS)
    az = np.random.normal(0, 0.1, TIMESTEPS)
    gx = np.random.normal(0, 0.05, TIMESTEPS)
    gy = np.random.normal(0, 0.05, TIMESTEPS)
    gz = np.random.normal(0, 0.05, TIMESTEPS)
    return np.concatenate([ax, ay, az, gx, gy, gz]).tolist()

def generate_punch():
    ax = np.random.normal(0, 0.1, TIMESTEPS)
    ay = np.random.normal(0, 0.1, TIMESTEPS)
    az = np.zeros(TIMESTEPS)
    spike_pos = np.random.randint(10, 30)
    az[spike_pos:spike_pos+5] = np.random.uniform(2.5, 3.5)
    az += np.random.normal(0, 0.1, TIMESTEPS)
    gx = np.random.normal(0, 0.05, TIMESTEPS)
    gy = np.random.normal(0, 0.05, TIMESTEPS)
    gz = np.random.normal(0, 0.05, TIMESTEPS)
    return np.concatenate([ax, ay, az, gx, gy, gz]).tolist()

def generate_circle():
    t = np.linspace(0, 2 * np.pi, TIMESTEPS)
    ax = np.sin(t) + np.random.normal(0, 0.1, TIMESTEPS)
    ay = np.cos(t) + np.random.normal(0, 0.1, TIMESTEPS)
    az = np.random.normal(0, 0.1, TIMESTEPS)
    gx = np.cos(t) * 0.5 + np.random.normal(0, 0.05, TIMESTEPS)
    gy = -np.sin(t) * 0.5 + np.random.normal(0, 0.05, TIMESTEPS)
    gz = np.random.normal(0, 0.05, TIMESTEPS)
    return np.concatenate([ax, ay, az, gx, gy, gz]).tolist()

gestures = {
    "wave":   generate_wave,
    "punch":  generate_punch,
    "circle": generate_circle
}

print("ESP32 Gesture Simulator started")
print(f"API endpoint: {API_URL}")
print(f"Device ID:    {DEVICE_ID}")
print("-" * 50)

for i in range(10):
    gesture_name = random.choice(list(gestures.keys()))
    sensor_data  = gestures[gesture_name]()

    payload = {
        "device_id":   DEVICE_ID,
        "sensor_data": sensor_data
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        result   = response.json()
        ts       = datetime.utcnow().strftime("%H:%M:%S")

        if response.status_code == 200:
            print(f"[{ts}] Sent: {gesture_name:8s} -> "
                  f"Predicted: {result['gesture']:8s} "
                  f"(confidence: {result['confidence']:.2%})")
        else:
            print(f"[{ts}] Error: {result}")

    except Exception as e:
        print(f"Request failed: {e}")

    time.sleep(1)

print("-" * 50)
print("Simulation complete — check DynamoDB for stored predictions!")
