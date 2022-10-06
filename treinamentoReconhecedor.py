import os

import cv2
import os
import numpy as np

# Carregar o xml do Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

captura = cv2.VideoCapture(0)

# Reconhecedor:
reconhecedor = cv2.face.LBPHFaceRecognizer_create()

# Variáveis:
trained = False
persons = os.listdir('train')

def trainData():  # Percoore todas as pastas/pessoas do diretorio para treinar o modelo
    global reconhecedor
    global trained
    trained = True
    persons = os.listdir('train')
    ids = []
    faces = []
    for i, p in enumerate(persons):
        for f in os.listdir(f'train/{p}'):
            img = cv2.imread(f'train/{p}/{f}', 0)
            faces.append(img)
            ids.append(i)
    reconhecedor.train(faces, np.array(ids))
    print("Treino finalizado!")


while True:

    verificador, frame = captura.read()
    if not verificador:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converter para cinza
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        roi = gray[y:y + h, x:x + w]  # Cortar apenas a face
        roi = cv2.resize(roi, (160, 160))

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)  # Desenhar o retângulo na face

        if trained:
            idp, confiabilidade = reconhecedor.predict(roi)
            if confiabilidade < 55:
                namePerson = persons[idp]
            else:
                namePerson = 'Desconhecido'

            cv2.putText(frame, namePerson, (x, y - 10), 1, 2, (255, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)

    # Fechar o while:
    if key == ord('q'):
        break

    if key == ord('t'):
        trainData()

captura.release()
cv2.destroyAllWindows()