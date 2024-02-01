import cv2
import numpy as np

def loadLabelsCOCO(path="Recursos/coco.names", sep='\n'):
    names = []
    with open(path, 'r') as file:
        names = [line.strip() for line in file if line.strip()]
    return names

def perform_object_detection(image):
    # Configuración
    IMG_WIDTH = 640.0
    IMG_HEIGHT = 640.0
    CLASS_PROBABILITY = 0.7
    NMS_THRESHOLD = 0.5
    CONFIDENCE_THRESHOLD = 0.5
    NUMBER_OF_OUTPUTS = 12

    # Carga las etiquetas COCO
    coco_labels_path = "Recursos/coco.names"
    coco_labels = loadLabelsCOCO(coco_labels_path)

    # Carga el modelo ONNX
    onnx_model_path = "Recursos/ray.onnx"
    neural_network = cv2.dnn.readNet(onnx_model_path)

    # Preprocesamiento de la imagen
    blob = cv2.dnn.blobFromImage(image, 1.0 / 255.0, (int(IMG_WIDTH), int(IMG_HEIGHT)), swapRB=True, crop=False)
    neural_network.setInput(blob)

    # Realiza la inferencia
    detections = neural_network.forward(neural_network.getUnconnectedOutLayersNames())
    print("Número de detecciones:", len(detections))
    # Postprocesamiento y dibujo de las detecciones en la imagen
    result_image = image.copy()
    height, width, _ = result_image.shape

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)

        # Verifica que class_id esté dentro de los límites de coco_labels
            if 0 <= class_id < len(coco_labels):
                confidence = scores[class_id]

                if confidence > CONFIDENCE_THRESHOLD:
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    print(f"Detección: Clase {coco_labels[class_id]}, Confianza: {confidence}")
                # Dibuja el cuadro y la etiqueta en la imagen resultante
                    color = (0, 255, 0)  # Color verde
                    cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
                    print(f"Detección: Clase {coco_labels[class_id]}, Confianza: {confidence}")
                # Usa la clase original (class_id) en lugar de np.argmax(scores)
                    label = f"{coco_labels[class_id]}: {confidence:.2f}"
                    cv2.putText(result_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return result_image
