import json
import boto3
import pickle
import numpy as np
import os
import uuid
from datetime import datetime

s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

BUCKET = 'esp32-tinyml-gestures-bucket'
TABLE  = 'GesturePredictions'

model = None
label_encoder = None

def load_models():
    global model, label_encoder
    if model is None:
        s3.download_file(BUCKET, 'models/model.pkl', '/tmp/model.pkl')
        s3.download_file(BUCKET, 'models/label_encoder.pkl', '/tmp/label_encoder.pkl')
        with open('/tmp/model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('/tmp/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        print("Models loaded from S3")

def lambda_handler(event, context):
    try:
        load_models()
        body = json.loads(event['body']) if isinstance(event.get('body'), str) else event
        sensor_data = body.get('sensor_data', [])
        device_id   = body.get('device_id', 'esp32-unknown')

        if not sensor_data:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No sensor_data provided'})
            }

        X = np.array(sensor_data).reshape(1, -1)
        prediction_encoded = model.predict(X)[0]
        prediction_proba   = model.predict_proba(X)[0]
        gesture = label_encoder.inverse_transform([prediction_encoded])[0]
        confidence = float(np.max(prediction_proba))

        table = dynamodb.Table(TABLE)
        item = {
            'prediction_id': str(uuid.uuid4()),
            'device_id':     device_id,
            'gesture':       gesture,
            'confidence':    str(round(confidence, 4)),
            'timestamp':     datetime.utcnow().isoformat(),
            'data_points':   str(len(sensor_data))
        }
        table.put_item(Item=item)

        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({
                'gesture':    gesture,
                'confidence': round(confidence, 4),
                'device_id':  device_id,
                'timestamp':  item['timestamp']
            })
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
