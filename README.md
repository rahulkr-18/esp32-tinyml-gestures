# ESP32 TinyML Gesture Detection 🚀

[![Python CI](https://github.com/rahulkr-18/esp32-tinyml-gestures/actions/workflows/ci-python.yml/badge.svg)](https://github.com/rahulkr-18/esp32-tinyml-gestures/actions/workflows/ci-python.yml)
[![Firmware CI](https://github.com/rahulkr-18/esp32-tinyml-gestures/actions/workflows/ci-firmware.yml/badge.svg)](https://github.com/rahulkr-18/esp32-tinyml-gestures/actions/workflows/ci-firmware.yml)

> Development mode: sensor data is synthetically generated matching real MPU6050 output characteristics. Firmware validated against ESP32 memory and size constraints via CI/CD.

## What this project does
Detects 3 hand gestures (wave, punch, circle) using ML models trained on simulated MPU6050 accelerometer data, with full AWS cloud integration.

## Architecture

## ML Models

| Model | Accuracy | Size | ESP32 Compatible |
|---|---|---|---|
| Random Forest | 100% | ~2MB pkl | Via MicroPython |
| TensorFlow Lite | 100% | 54 KB | Direct firmware flash |

## Tech Stack
- Python 3.11, scikit-learn 1.5.0, TensorFlow 2.16
- AWS Lambda, API Gateway, DynamoDB, S3, IoT Core
- GitHub Actions CI/CD — auto-train + auto-deploy on every push
- ESP-IDF + MicroPython firmware (CI-validated)

## Project Structure

| Folder | Contents |
|---|---|
| .github/workflows/ | CI/CD pipelines |
| firmware_micropython/ | MicroPython gesture code |
| firmware_tflite/ | TFLite C++ firmware + model header |
| python_training/ | Training + conversion scripts |
| data/raw/ | Synthetic sensor data CSV files |
| data/processed/ | Trained models + TFLite file |
| docs/ | Confusion matrix + feature plots |

## Performance

| Metric | Value |
|---|---|
| Gesture classes | wave, punch, circle |
| Training samples | 3000 (1000 per class) |
| Features | 300 (50 timesteps x 6 axes) |
| Random Forest accuracy | 100% |
| TFLite accuracy | 100% |
| TFLite model size | 54 KB |
| ESP32 flash limit | 1024 KB |
| Cloud inference latency | ~1 second |
| On-device inference | < 10ms |

## Quick Start

git clone https://github.com/rahulkr-18/esp32-tinyml-gestures.git
cd esp32-tinyml-gestures
pip install -r requirements.txt
python python_training/generate_data.py
python python_training/train_model.py
python python_training/esp32_simulator.py

## CI/CD Pipeline
Every push automatically:
1. Generates fresh training data
2. Trains and validates model (accuracy threshold 85%)
3. Converts to TFLite and checks ESP32 size limit
4. Uploads model artifacts to AWS S3
5. Saves downloadable build artifacts on GitHub

## AWS Services Used (all Free Tier)

| Service | Purpose |
|---|---|
| S3 | Model file storage |
| Lambda | Serverless inference endpoint |
| API Gateway | Public REST API |
| DynamoDB | Prediction result logging |
| IoT Core | ESP32 device simulation |
