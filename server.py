from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import uvicorn
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🔄 Modeller Yükleniyor... Lütfen bekleyin.")
model_eye = YOLO("/home/murat/Desktop/bitirme/models/yolov8_fruit_v13.pt")   
model_brain = YOLO("/home/murat/Desktop/bitirme/models/final_model.pt")      
print("✅ Modeller Hazır! API Başlatılıyor...")

@app.get("/")
async def health_check():
    return {"status": "online"}

def smart_predict_process(img_array):
    results_brain = model_brain.predict(img_array, conf=0.25, verbose=False)[0]
    results_eye = model_eye.predict(img_array, conf=0.25, verbose=False)[0]
    
    final_detections = []
    matched_brain_indices = set()

    for i, box_eye in enumerate(results_eye.boxes.xyxy.cpu().numpy()):
        final_label = "Nesne"
        final_color = (200, 200, 200)
        
        if len(results_brain.boxes) > 0:
            best_iou = 0
            best_idx = -1
            for j, box_brain in enumerate(results_brain.boxes.xyxy.cpu().numpy()):
                x1 = max(box_eye[0], box_brain[0]); y1 = max(box_eye[1], box_brain[1])
                x2 = min(box_eye[2], box_brain[2]); y2 = min(box_eye[3], box_brain[3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area1 = (box_eye[2] - box_eye[0]) * (box_eye[3] - box_eye[1])
                area2 = (box_brain[2] - box_brain[0]) * (box_brain[3] - box_brain[1])
                iou = inter / (area1 + area2 - inter + 1e-6)
                
                if iou > 0.3 and iou > best_iou:
                    best_iou = iou
                    best_idx = j
            
            if best_idx != -1:
                class_id = int(results_brain.boxes.cls[best_idx])
                final_label = results_brain.names[class_id]
                matched_brain_indices.add(best_idx)
                
                if "Fresh" in final_label: final_color = (0, 255, 0)
                elif "Rotten" in final_label: final_color = (0, 0, 255)
                else: final_color = (255, 0, 0)
        
        final_detections.append((box_eye, final_label, final_color))

    for j, box_brain in enumerate(results_brain.boxes.xyxy.cpu().numpy()):
        if j not in matched_brain_indices:
            class_id = int(results_brain.boxes.cls[j])
            label = results_brain.names[class_id]
            if "Fresh" in label: color = (0, 255, 0)
            elif "Rotten" in label: color = (0, 0, 255)
            else: color = (255, 0, 0)
            final_detections.append((box_brain, label, color))

    plot_img = img_array.copy()
    for box, label, color in final_detections:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(plot_img, (x1, y1), (x2, y2), color, 3)
        
        text_scale = 0.8 if label != "Nesne" else 0.5
        thickness = 2 if label != "Nesne" else 1
        
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness)
        cv2.rectangle(plot_img, (x1, y1 - 25), (x1 + w, y1), color, -1)
        cv2.putText(plot_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), thickness)

    return plot_img

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    processed_img = smart_predict_process(img)
    
    _, encoded_img = cv2.imencode('.jpg', processed_img)
    base64_str = base64.b64encode(encoded_img).decode('utf-8')
    
    return {"image": base64_str}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)