import numpy as np
import cv2 as cv
import time

slider_max = 100

def nothing(x):
    pass

ini = time.time()

#Realiza a leitura da imagem
imgcolor = cv.imread("robos.jpg", 1)

#Altera o tamanho da imagem
#imgcolor = cv.resize(imgcolor, None, fx=0.2, fy=0.2, interpolation = cv.INTER_CUBIC)

#Cria uma Trackbar para controlar o valor das variaveis H,S,V
cv.namedWindow('image')
cv.createTrackbar('h', 'image', 0, 255, nothing)
cv.createTrackbar('s', 'image', 0, 255, nothing)
cv.createTrackbar('v', 'image', 0, 255, nothing)


black = np.uint8([[[0,0,0 ]]])
hsv_black = cv.cvtColor(black,cv.COLOR_BGR2HSV)

#Cria um kernel para as funções de filtro
kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
# do the laplacian filtering as it is
# well, we need to convert everything in something more deeper then CV_8U
# because the kernel has some negative values,
# and we can expect in general to have a Laplacian image with negative values
# BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
# so the possible negative number will be truncated
imgLaplacian = cv.filter2D(imgcolor, cv.CV_32F, kernel)
sharp = np.float32(imgcolor)
imgResult = sharp - imgLaplacian

# convert back to 8bits gray scale
imgResult = np.clip(imgResult, 0, 255)
imgcolor = imgResult.astype('uint8')
imgLaplacian = np.clip(imgLaplacian, 0, 255)
imgLaplacian = np.uint8(imgLaplacian)

#Inicializa as variaveis H,S,V
h = 50
s = 50
v = 50
cont = 0;
#cap = cv.VideoCapture(0)

while (1):
    #_, frame = cap.read()
    #Transforma a imagem lida para cinza
    gray = cv.cvtColor(imgcolor, cv.COLOR_BGR2GRAY)
    #Realiza um threshold na imagem cinza
    ret, thresh = cv.threshold(gray, 30, 255, cv.THRESH_BINARY_INV)
    #Cria uma imagem HSV da imagem lida
    hsv = cv.cvtColor(imgcolor, cv.COLOR_RGB2HSV)
    #Cria os intervalos do HSV para a criação da mascara
    lower_black = np.array ([0,0, 0])
    upper_black = np.array ([h,s,v])
    mask = cv.inRange(hsv, lower_black, upper_black)                    

    #dist_transform = cv.distanceTransform(thresh, cv.DIST_L2, 5)
    #ret, sure_fg = cv.threshold(dist_transform, 0.7*dist_transform.max(), 255,0)

    #Cria um novo kernel para fazer a dilatação dos contornos
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7))

    #Realiza a função Close (Dilatação e Erosão)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations = 6)

    cv.imshow("mask", mask) 
    
    labels = cv.connectedComponentsWithStats(mask, 8, cv.CC_STAT_AREA, cv.CV_32S)
    num_labels = labels[0]
    centroids = labels[3]
    stats = labels[2]

    robos = np.zeros((10,10))
    #contours,hierarchy = cv.findContours(thresh, 1, 2)
    print(centroids)
    #Encontra os contornos na Mascara
    _, contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    if (cont ==0):
        #Desenha os contornos dentro de uma copia da imagem original
        img=imgcolor
        cv.drawContours(img, contours, -1, (0,255,0), 3)
        cont+=1    
    
    cv.imshow("img", img)
    h = cv.getTrackbarPos('h', 'image')
    s = cv.getTrackbarPos('s', 'image')
    v = cv.getTrackbarPos('s', 'image')
    key = cv.waitKey(1)
    if key == 27:
        break

#cv2.CC_STAT_AREA The total area (in pixels) of the connected component

    
cap.release()    
cv.imshow("thresh", thresh)
cv.destroyAllWindows()
