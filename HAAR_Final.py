import cv2
from matplotlib import pyplot as plt

car_cascade = cv2.CascadeClassifier('cars.xml') #load the cascade classifier
img = cv2.imread('car2.jpg', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to greyscale
plt.imshow(gray)
cv2.imshow('grey',gray)
# Detect cars
cars = car_cascade.detectMultiScale(gray, 1.1, 1) # has all the
                                    #recatangles of detected cars and has a
                                    #numoy array woith x,y,w,h

print(cars)
# Draw border
ncars=0
for (x, y, w, h) in cars:
    if(w<70 and h<70):
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2) #topleft,bottomright,color,thickness
        ncars = ncars + 1
        crop_img = img[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (200, 250)) 
        cv2.imwrite(str(ncars)+'.png',resized_img)

    else:
        
        cv2.rectangle(img, (x,y), (x+w-50,y+h-50), (255,0,0), 2) #topleft,bottomright,color,thickness
        ncars = ncars + 1
        crop_img = img[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (300, 350)) 
        cv2.imwrite(str(ncars)+'.png',resized_img)
# Show image
plt.figure(figsize=(10,20))
plt.imshow(img)
cv2.imshow('car',img)
cv2.waitKey(0)
