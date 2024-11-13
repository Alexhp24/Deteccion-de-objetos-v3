import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Cargar el modelo YOLOv8 preentrenado
model = YOLO('yolov8n.pt')  # Puedes usar cualquier variante de YOLOv8

# Diccionario de traducción
translation_dict = {
    "person": "persona",
    "bicycle": "bicicleta",
    "car": "coche",
    "motorcycle": "motocicleta",
    "airplane": "avión",
    "bus": "autobús",
    "train": "tren",
    "truck": "camión",
    "boat": "barco",
    "traffic light": "semáforo",
    "fire hydrant": "hidrante",
    "stop sign": "señal de alto",
    "parking meter": "parquímetro",
    "bench": "banco",
    "bird": "pájaro",
    "cat": "gato",
    "dog": "perro",
    "horse": "caballo",
    "sheep": "oveja",
    "cow": "vaca",
    "elephant": "elefante",
    "bear": "oso",
    "zebra": "cebra",
    "giraffe": "jirafa",
    "backpack": "mochila",
    "umbrella": "paraguas",
    "handbag": "bolso",
    "tie": "corbata",
    "suitcase": "maleta",
    "frisbee": "frisbee",
    "skis": "esquís",
    "snowboard": "tabla de snowboard",
    "sports ball": "pelota",
    "kite": "cometa",
    "baseball bat": "bate de béisbol",
    "baseball glove": "guante de béisbol",
    "skateboard": "patineta",
    "surfboard": "tabla de surf",
    "tennis racket": "raqueta de tenis",
    "bottle": "botella",
    "wine glass": "copa de vino",
    "cup": "taza",
    "fork": "tenedor",
    "knife": "cuchillo",
    "spoon": "cuchara",
    "bowl": "bol",
    "banana": "banana",
    "apple": "manzana",
    "sandwich": "sándwich",
    "orange": "naranja",
    "broccoli": "brócoli",
    "carrot": "zanahoria",
    "pizza": "pizza",
    "donut": "dona",
    "cake": "pastel",
    "chair": "silla",
    "couch": "sofá",
    "potted plant": "planta en maceta",
    "bed": "cama",
    "dining table": "mesa de comedor",
    "toilet": "inodoro",
    "tv": "televisor",
    "laptop": "portátil",
    "mouse": "ratón",
    "remote": "control remoto",
    "keyboard": "teclado",
    "cell phone": "celular",
    "microwave": "microondas",
    "oven": "horno",
    "toaster": "tostadora",
    "sink": "fregadero",
    "refrigerator": "refrigerador",
    "book": "libro",
    "clock": "reloj",
    "vase": "jarrón",
    "scissors": "tijeras",
    "teddy bear": "oso de peluche",
    "hair drier": "secador de pelo",
    "toothbrush": "cepillo de dientes"
}

# Iniciar la captura de video de la cámara
cap = cv2.VideoCapture(0)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar detección de objetos
    results = model(frame)

    # Dibujar las cajas y etiquetas en el frame
    for r in results[0].boxes:
        x1, y1, x2, y2 = r.xyxy[0]
        confidence = r.conf[0]
        class_id = int(r.cls[0])
        label = model.names[class_id]
        translated_label = translation_dict.get(label, label)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{translated_label} {confidence:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convertir BGR a RGB para mostrar con Matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Mostrar el frame con Matplotlib
    ax.clear()
    ax.imshow(frame_rgb)
    plt.pause(0.001)

    # Agregar un mecanismo para salir del bucle
    if plt.get_fignums() == []:  # Si la figura se ha cerrado, romper el bucle
        break

cap.release()
plt.close()