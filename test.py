import cv2

cap = cv2.VideoCapture("output.mov")
frameCount = 1
while 1:
    ret , frame = cap.read()
    if ret==False:
        print("false")
        break
    print(frameCount)
    frameCount+=1
    cv2.imshow("f", frame)
    cv2.waitKey(5)
