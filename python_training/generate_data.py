import numpy as np
import pandas as pd
import os

np.random.seed(42)
SAMPLES_PER_GESTURE = 1000
TIMESTEPS = 50
OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_wave(n_samples, timesteps):
    data = []
    for _ in range(n_samples):
        t = np.linspace(0, 2 * np.pi, timesteps)
        ax = np.sin(2 * t) + np.random.normal(0, 0.1, timesteps)
        ay = np.random.normal(0, 0.1, timesteps)
        az = np.random.normal(0, 0.1, timesteps)
        gx = np.random.normal(0, 0.05, timesteps)
        gy = np.random.normal(0, 0.05, timesteps)
        gz = np.random.normal(0, 0.05, timesteps)
        sample = np.concatenate([ax, ay, az, gx, gy, gz])
        data.append(sample)
    return np.array(data)

def generate_punch(n_samples, timesteps):
    data = []
    for _ in range(n_samples):
        ax = np.random.normal(0, 0.1, timesteps)
        ay = np.random.normal(0, 0.1, timesteps)
        az = np.zeros(timesteps)
        spike_pos = np.random.randint(10, 30)
        az[spike_pos:spike_pos+5] = np.random.uniform(2.5, 3.5)
        az += np.random.normal(0, 0.1, timesteps)
        gx = np.random.normal(0, 0.05, timesteps)
        gy = np.random.normal(0, 0.05, timesteps)
        gz = np.random.normal(0, 0.05, timesteps)
        sample = np.concatenate([ax, ay, az, gx, gy, gz])
        data.append(sample)
    return np.array(data)

def generate_circle(n_samples, timesteps):
    data = []
    for _ in range(n_samples):
        t = np.linspace(0, 2 * np.pi, timesteps)
        ax = np.sin(t) + np.random.normal(0, 0.1, timesteps)
        ay = np.cos(t) + np.random.normal(0, 0.1, timesteps)
        az = np.random.normal(0, 0.1, timesteps)
        gx = np.cos(t) * 0.5 + np.random.normal(0, 0.05, timesteps)
        gy = -np.sin(t) * 0.5 + np.random.normal(0, 0.05, timesteps)
        gz = np.random.normal(0, 0.05, timesteps)
        sample = np.concatenate([ax, ay, az, gx, gy, gz])
        data.append(sample)
    return np.array(data)

print("Generating gesture data...")

wave_data = generate_wave(SAMPLES_PER_GESTURE, TIMESTEPS)
punch_data = generate_punch(SAMPLES_PER_GESTURE, TIMESTEPS)
circle_data = generate_circle(SAMPLES_PER_GESTURE, TIMESTEPS)

wave_df = pd.DataFrame(wave_data)
wave_df["label"] = "wave"
wave_df.to_csv(f"{OUTPUT_DIR}/wave.csv", index=False)
print(f"wave.csv saved — {len(wave_df)} samples")

punch_df = pd.DataFrame(punch_data)
punch_df["label"] = "punch"
punch_df.to_csv(f"{OUTPUT_DIR}/punch.csv", index=False)
print(f"punch.csv saved — {len(punch_df)} samples")

circle_df = pd.DataFrame(circle_data)
circle_df["label"] = "circle"
circle_df.to_csv(f"{OUTPUT_DIR}/circle.csv", index=False)
print(f"circle.csv saved — {len(circle_df)} samples")

all_data = pd.concat([wave_df, punch_df, circle_df], ignore_index=True)
all_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)
all_data.to_csv(f"{OUTPUT_DIR}/gestures_all.csv", index=False)
print(f"gestures_all.csv saved — {len(all_data)} total samples")

print("\nData generation complete!")
print(f"Gestures: wave={len(wave_df)}, punch={len(punch_df)}, circle={len(circle_df)}")
print(f"Features per sample: {wave_data.shape[1]} (50 timesteps x 6 axes)")
print(f"Files saved to: {OUTPUT_DIR}")
