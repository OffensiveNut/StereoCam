import cv2

cap = cv2.VideoCapture(2)
cap2 = cv2.VideoCapture(0)

num = 0

while cap.isOpened() and cap2.isOpened():
    
    success1, img= cap.read()
    success2, img2= cap2.read()
    
    k= cv2.waitKey(5)
    if k==ord('q'):
        break
    if k==ord('s'):
        cv2.imwrite('images/stereoLeft/imageL'+str(num) + '.png', img)
        cv2.imwrite('images/stereoRight/imageR'+str(num) + '.png', img2)
        print('image saved!')
        num += 1
        
    cv2.imshow('Left Camera', img)
    cv2.imshow('Right Camera', img2)