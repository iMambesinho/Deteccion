import torch
import cv2
import os

# Cargar modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
model.eval()

vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']
image_path = 'estacionamiento.jpg'

# Verificar si existe la imagen
if not os.path.exists(image_path):
    raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = model(img_rgb)
detections = results.pandas().xyxy[0]

for _, row in detections.iterrows():
    if row['name'] in vehicle_classes:
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        label = f"{row['name']} {row['confidence']:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Mostrar imagen usando OpenCV
cv2.imshow("Vehículos detectados", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
