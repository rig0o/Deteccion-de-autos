import cv2

cap = cv2.VideoCapture('urs.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)

print(fps)

while True:
    ret, frame = cap.read()
    if ret == False:break
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 1000, 500)
    
    cv2.imshow('frame',frame)
    k = cv2.waitKey(1)&0xFF
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()