
import cv2
import numpy as np
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
img1 = cv2.imread(r"biscuit.jpg")

img= cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

'''COUNTING NUMBER OF SHAPES USING DICTIONARY'''
count = dict()
count['triangle'],count['rectangle'],count['square'],count['pentagon'],count['circle'],count['e'] = 0,0,0,0,0,0

'''THRESHOLDING GIVES BINARY IMAGE WHICH PROVIDES BETTER ACCURACY IN CONTOURING.
USED CONTOURS FOR DETECTING EDGES OF SHAPES. '''
_, threshold = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
contours,_ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

'''PLOTTING EACH CONTOUR AND DETERMING OBJECT SHAPE'''

for cnt in contours:
    area = cv2.contourArea(cnt)
    '''APPROXIMATES CURVES OF POLYGONS WITH PRECISION'''
    approx = cv2.approxPolyDP(cnt, 0.0147*cv2.arcLength(cnt, True), True)

    x = approx.ravel()[0]
    y = approx.ravel()[1]
    if area > 300:
        cv2.drawContours(img1, [approx], 0, (255, 255, 0), 5)

        '''CONDITIONS FOR DETERMINING SPECIFIC SHAPES'''
        if len(approx) == 3:
            cv2.putText(img1, "Triangle", (x, y), font, 1, (0))
            count['triangle']+=1
        elif len(approx) == 4:
            x1, y1, w, h = cv2.boundingRect(approx)
            aspectRatio = float(w) / h
            #print(aspectRatio,x1,y1,w,h)
            '''DEFERENTIATED BETWEEN RECTANGLE AND SQUARE USING ASPECT RATIO PROPERTY'''
            if (aspectRatio >= 0.95 and aspectRatio <= 1.7):
                cv2.putText(img1, "Square", (x, y), font, 1, (0))
                count['square'] += 1
            else:
                cv2.putText(img1, "Rectangle", (x, y), font, 1, (0))
                count['rectangle'] += 1
        elif len(approx) == 5:
            cv2.putText(img1, "Pentagon", (x, y), font, 1, (0))
            count['pentagon'] += 1
        elif 6 < len(approx) < 15:
            cv2.putText(img1, "Ellipse", (x, y), font, 1, (0))
            count['e'] += 1
        else:
            cv2.putText(img1, "Circle", (x, y), font, 1, (0))
            count['circle'] += 1

cv2.putText(img1, "COUNT : ", (5,25), font, 1, (150,150,150))
cv2.putText(img1, "Triangle : "+str(count['triangle']), (5,45), font, 1, (150,150,150))
cv2.putText(img1, "Rectangle : "+str(count['rectangle']), (5,65), font, 1, (150,150,150))
cv2.putText(img1, "Square : "+str(count['square']), (5,85), font, 1, (150,150,150))
cv2.putText(img1, "Circle : "+str(count['circle']), (5,105), font, 1, (150,150,150))
cv2.putText(img1, "Ellipse : "+str(count['e']), (5,125), font, 1, (150,150,150))
cv2.putText(img1, "Pentagon : "+str(count['pentagon']), (5,145), font, 1, (150,150,150))

print("triangle", count['triangle'])
print("square", count['square'])
print("rectangle", count['rectangle'])
print("circle", count['circle'])
print("ellipse", count['e'])
print("pentagon", count['pentagon'])

cv2.imshow("Shapes", img1)
cv2.imshow("Threshold", threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()