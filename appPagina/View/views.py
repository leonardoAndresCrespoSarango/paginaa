import io
from PIL import Image as im

import numpy as np
from django.shortcuts import render
from django.views.generic.edit import CreateView
import cv2
from appPagina.Logica.modeloSNN import perform_object_detection  
from django.http import HttpResponse
from appPagina.Logica.utils import build_model, detect, load_classes, wrap_detection, class_list
import base64

def deteccion(request):
    if request.method == 'POST' and request.FILES['imagen']:
        imagen = request.FILES['imagen']

        # Utiliza directamente las funciones de detección
        net = build_model(is_cuda=False)  # Ajusta el valor de is_cuda según tus necesidades
        image = cv2.imdecode(np.fromstring(imagen.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 640))  # Ajusta el tamaño según tus necesidades
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (640, 640), swapRB=True, crop=False)
        net.setInput(blob)
        preds = net.forward()
        class_ids, confidences, boxes = wrap_detection(image, preds[0])

        # Aquí puedes definir 'resultado' con la información que quieras mostrar
        resultado = {'class_ids': class_ids, 'confidences': confidences, 'boxes': boxes}

        # Convertir la imagen con las detecciones a base64
        image_with_detections = image.copy()
        for i in range(len(boxes)):
            box = boxes[i]
            class_id = class_ids[i]
            cv2.rectangle(image_with_detections, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
            cv2.putText(image_with_detections, class_list[class_id], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        _, buffer = cv2.imencode('.png', image_with_detections)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        resultado['imagen'] = image_base64
        # ... Resto del código para mostrar resultados en la plantilla ...
        return render(request, 'page.html', {'resultado': resultado})

    return render(request, 'page.html')


def base(request):
    return render(request, 'base.html')