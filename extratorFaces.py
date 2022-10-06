from mtcnn import MTCNN #Reconhece as faces
from PIL import Image  #Manipula imagens
from os import listdir  #Lista diretório
from os.path import isdir  #Reconhece se é um diretório
from numpy import asarray  #Converter uma img em pillow para numpy

detector = MTCNN()  #Encontrar as faces dentro da img


def extrair_face(arquivo, size=(160, 160)):

    img = Image.open(arquivo)  #Passa o caminho do arquivo
    img = img.convert('RGB')  #Converter a img em RGB
    array = asarray(img)  #Converte a img para matriz, pois o detector não ler um arquivo pillow
    results = detector.detect_faces(array)
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    face = array[y1:y2, x1:x2]  #Extrai apenas o rosto da img original
    image = Image.fromarray(face)  #Converteu para Pillow novamente
    image = image.resize(size)  #Redimensionar

    return image


#Carregar os diretórios dentro da pasta fotos:
def load_dir(diretorio_src, diretorio_target):  #diretorio_src: fotos   diretorio_target: train
    for subdir in listdir(diretorio_src):
        path = diretorio_src + subdir + '\\'  #Pegar o caminho do subdiretório
        path_tg = diretorio_target + subdir + '\\'

        #Testar se o que está sendo lido é um diretório ou arquivo:
        if not isdir(path):
            continue

        load_fotos(path, path_tg)

def load_fotos(diretorio_src, diretorio_target):
    print(diretorio_src)
    print(diretorio_target)

if __name__ == '__main__':
    load_dir("/pythonProject/fotos\\", "C:\\Users\\DELL\\PycharmProjects\\pythonProject\\train\\")
