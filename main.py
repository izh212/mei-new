from fastapi import FastAPI, WebSocket
import numpy as np
import io
from PIL import Image
from ultralytics import YOLO

# Load YOLO model
model = YOLO("./best_int8_openvino_model", task='detect')

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            # Receive raw JPEG bytes from client
            data = await websocket.receive_bytes()

            # Decode and convert to numpy array
            image = Image.open(io.BytesIO(data)).convert("RGB")
            img_np = np.array(image)

            # Inference
            results = model(img_np, verbose=False)[0]

            # Get absolute pixel bbox coordinates (xyxy)
            boxes = results.boxes.xyxy.cpu().numpy().astype(np.float32)  # shape: (N, 4)

            # Send raw float32 bounding box bytes
            await websocket.send_bytes(boxes.tobytes())

        except Exception as e:
            print(f"Error: {e}")
            break
