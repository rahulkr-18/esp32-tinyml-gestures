# ESP32 TinyML Gesture Detection 🚀

[![Python CI](https://github.com/rahulkr-18/esp32-tinyml-gestures/actions/workflows/ci-python.yml/badge.svg)](https://github.com/rahulkr-18/esp32-tinyml-gestures/actions/workflows/ci-python.yml)
[![Firmware CI](https://github.com/rahulkr-18/esp32-tinyml-gestures/actions/workflows/ci-firmware.yml/badge.svg)](https://github.com/rahulkr-18/esp32-tinyml-gestures/actions/workflows/ci-firmware.yml)

ESP32 gesture detection using TinyML — simulated sensor data + AWS cloud pipeline.

## What this project does
- Simulates MPU6050 accelerometer gesture data (no hardware needed)
- Trains ML models: Random Forest + TensorFlow Lite
- Sends predictions to AWS IoT Core via MQTT
- Stores results in DynamoDB
- Exposes REST API via Lambda + API Gateway

## Stack
- Python 3.11, scikit-learn, TensorFlow
- AWS IoT Core, Lambda, DynamoDB, S3, API Gateway
- GitHub Actions CI/CD
- ESP32 firmware (ESP-IDF + MicroPython)

## Status
🔧 In progress — Phase 1: Environment setup
