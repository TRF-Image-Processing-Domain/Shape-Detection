
'''shape detection using thresholding'''
import cv2

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

'''COUNTING NUMBER OF SHAPES USING DICTIONARY'''
count = dict()

cap = cv2.VideoCapture(0)
while 1:

    count['triangle'], count['rectangle'], count['square'], count['pentagon'], count['circle'] = 0, 0, 0, 0, 0
    ret, img = cap.read()
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    '''THRESHOLDING GIVES BINARY IMAGE WHICH PROVIDES BETTER ACCURACY IN CONTOURING.
    USED CONTOURS FOR DETECTING EDGES OF SHAPES. '''
    _, threshold = cv2.threshold(frame, 175, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    '''PLOTTING EACH CONTOUR AND DETERMINING OBJECT SHAPE'''
    for cnt in contours:
        area = cv2.contourArea(cnt)
        '''APPROXIMATES CURVES OF POLYGONS WITH PRECISION'''
        approx = cv2.approxPolyDP(cnt, 0.017 * cv2.arcLength(cnt, True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        if area > 800:
            cv2.drawContours(frame, [approx], 0, (0, 0, 0), 3)
            '''CONDITIONS FOR DETERMINING SPECIFIC SHAPES'''
            if len(approx) == 3:
                cv2.putText(img, "Triangle", (x, y), font, 1, 0)
                count['triangle'] += 1
            elif len(approx) == 4:
                x1, y1, w, h = cv2.boundingRect(approx)
                aspectRatio = float(w) / h
                '''DEFERENTIATED BETWEEN RECTANGLE AND SQUARE USING ASPECT RATIO PROPERTY'''
                if 0.95 <= aspectRatio <= 1.7:
                    cv2.putText(img, "Square", (x, y), font, 1, 0)
                    count['square'] += 1
                else:
                    cv2.putText(img, "Rectangle", (x, y), font, 1, 0)
                    count['rectangle'] += 1
            elif len(approx) == 5:
                cv2.putText(img, "Pentagon", (x, y), font, 1, 0)
                count['pentagon'] += 1
            elif 6 < len(approx) < 20:
                cv2.putText(img, "Circle", (x, y), font, 1, 0)
                count['circle'] += 1
            else:
                cv2.putText(img, "Polygon", (0, 0), font, 1, 0)

    cv2.imshow("shapes", img)
    cv2.imshow("threshold", threshold)
    k = cv2.waitKey(1)
    if k == 32:
        break

    print("Count for shapes : ")
    print("Triangle : ", count['triangle'])
    print("Square : ", count['square'])
    print("Rectangle : ", count['rectangle'])
    print("Pentagon : ", count['pentagon'])
    print("Circle : ", count['circle'])

cap.release()
cv2.destroyAllWindows()
