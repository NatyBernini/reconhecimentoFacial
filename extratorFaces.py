'''
Esta aplicação irá criar um diretório, caso ele não exista, com o nome da pessoa.
Depois ele irá extrair a face da webcam quantas vezes o usuário apertar a tecla f
'''

# Importes:
import cv2
import os  # Lib para diretórios
import time

# Ler a webcam:
captura = cv2.VideoCapture(0)

# Carregar o xml do Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# Irá salvar o nome da pessoa:
def savePerson():
    global identificacao
    print('Qual o seu nome: ')
    name = input()
    identificacao = name


# Criação dos diretórios:
def saveDir(img):
    global identificacao
    id = time.strftime('%Y%m%d-%H%M%S')
    if not os.path.exists('train'):  # Cria a pasta train para salvar os diretórios de cada pessoa
        os.makedirs('train')
    if not os.path.exists(f'train/{identificacao}'):  # Cria as subpastas para cada pessoa
        os.makedirs(f'train/{identificacao}')
    files = os.listdir(f'train/{identificacao}')
    cv2.imwrite(f'train/{identificacao}/{str(id)}.jpg', img)


# Variáveis:
identificacao = ''


while True:

    verificador, frame = captura.read()
    if not verificador:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converter para cinza
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    key = cv2.waitKey(1)

    # Fechar o while:
    if key == ord('q'):
        break

    # Salvar imagens:
    if key == ord('s'):
        savePerson()

    for (x, y, w, h) in faces:

        roi = gray[y:y+h, x:x+w]  # Cortar apenas a face

        cv2.rectangle(frame, (x,y), (x+w, y+h), (200, 0, 0), 3)  # Desenhar o retângulo na face

        if cv2.waitKey(1) == ord('f'):
            saveDir(roi)

    cv2.imshow('frame', frame)


captura.release()
cv2.destroyAllWindows()
