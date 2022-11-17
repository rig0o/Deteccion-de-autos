import cv2
import numpy as np
from datetime import datetime


cap = cv2.VideoCapture('urs.mp4')


fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))


cuadro = 0
while True:

    ret, frame = cap.read()
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 1000, 500)
    #cv2.namedWindow('recorte', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('recorte', 1000, 500)
    #Puntos donde se buscara
    area_pts = np.array([[900, 1650],[1580,1650],[1510,2100],[750,2100]])
    color = (0, 255, 0)

    # Region
    cv2.drawContours(frame, [area_pts],-1,color,10)
    texto_estado = "Estado: No se ha detectado auto"
    cv2.rectangle(frame,(0,0),(frame.shape[1],100),(0,0,0),-1)

    if ret == False:break
    if cuadro == 5:
        cuadro = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        # recorte
        recorte = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
        recorte = cv2.drawContours(recorte, [area_pts],-1,(255),-1)
        img_area = cv2.bitwise_and(gray,gray,mask=recorte)

        #Obtendremos la imagen binaria
        fgmask = fgbg.apply(img_area)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.dilate(fgmask, None, iterations=2)

        #Filtra contornes segun el area
        cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        path = '/home/rodrigo/Workspace/auto_bueno/output'
        for cnt in cnts:
            if cv2.contourArea(cnt) > 70000:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0), 2)
                save = frame[y:y+h, x:x+w]
                texto_estado = "Estado: Alerta Movimiento Detectado!"
                cv2.imwrite(f'{path}/plate{datetime.now()}.png',save)
                color = (0, 0, 255)    

        #Texto 
        cv2.putText(frame, texto_estado, (300,90),cv2.FONT_HERSHEY_SIMPLEX,2,color,10)
        cuadro += 1
    else:
        cv2.putText(frame, texto_estado, (300,90),cv2.FONT_HERSHEY_SIMPLEX,2,color,10)
        cuadro += 1
    cv2.imshow('frame',frame)
    #cv2.imshow('recorte',fgmask)

    k = cv2.waitKey(1)&0xFF
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()