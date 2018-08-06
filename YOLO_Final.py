import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
#%matplotlib inline
#% pylab inline

import keras
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense, Activation, Reshape
import struct
from PIL import Image

import scipy
import scipy.misc
import scipy.cluster
import codecs
import webcolors
from sklearn.cluster import KMeans
import utils

# Pre trained weights require this ordering
keras.backend.set_image_dim_ordering('th')

def get_model():
    model = Sequential()
    
    #padding
    #border_mode
    #Convolution2D
    
    # Layer 1
    model.add(Convolution2D(16, 3, 3,input_shape=(3,448,448),border_mode='same',subsample=(1,1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    
    # Layer 2
    model.add(Convolution2D(32,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    # Layer 3
    model.add(Convolution2D(64,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    
    # Layer 4
    model.add(Convolution2D(128,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    
    # Layer 5
    model.add(Convolution2D(256,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    
    # Layer 6
    model.add(Convolution2D(512,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    
    # Layer 7
    model.add(Convolution2D(1024,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    
    # Layer 8
    model.add(Convolution2D(1024,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    
    # Layer 9
    model.add(Convolution2D(1024,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    
    model.add(Flatten())
    
    # Layer 10
    model.add(Dense(256))
    
    # Layer 11
    model.add(Dense(4096))
    model.add(LeakyReLU(alpha=0.1))
    
    # Layer 12
    model.add(Dense(1470))
    
    return model

# Preprocessing

def crop_and_resize(image):
    #cropped = image[300:650,500:,:]
    #return cv2.resize(cropped, (448,448))
    resized=cv2.resize(image, (448,448))
    #plt.imshow(resized)
    #plt.show()
    return resized

def normalize(image):
    normalized = 2.0*image/255.0 - 1
    return normalized

def preprocess(image):
    cropped = crop_and_resize(image)
    normalized = normalize(cropped)
    #normalized = normalize(image)


    # The model works on (channel, height, width) ordering of dimensions
    transposed = np.transpose(normalized, (2,0,1))
    #plt.imshow(transposed)
    #plt.show()
    #print (transposed)
    #transposed = np.transpose(cropped, (2,0,1))
    return transposed

class Box:
    def __init__(self):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.prob = float()
        
def overlap(x1, w1, x2, w2):
    l1 = x1 - w1 / 2.
    l2 = x2 - w2 / 2.
    left = max(l1, l2)
    r1 = x1 + w1 / 2.
    r2 = x2 + w2 / 2.
    right = min(r1, r2)
    return right - left


def box_intersection(a, b):
    """

    :param a: Box 1
    :param b: Box 2
    :return: Intersection area of the 2 boxes
    """
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0
    area = w * h
    return area


def box_union(a, b):
    """

    :param a: Box 1
    :param b: Box 2
    :return: Area under the union of the 2 boxes
    """
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u


def box_iou(a, b):
    """

    :param a: Box 1
    :param b: Box 2
    :return: Intersection over union, which is ratio of intersection area to union area of the 2 boxes
    """
    return box_intersection(a, b) / box_union(a, b)



def yolo_output_to_car_boxes(yolo_output, threshold=0.2, sqrt=1.8, C=20, B=2, S=7):

    # Position for class 'car' in the VOC dataset classes
    car_class_number = 6

    boxes = []
    SS = S*S  # number of grid cells
    prob_size = SS*C  # class probabilities
    conf_size = SS*B  # confidences for each grid cell

    probabilities = yolo_output[0:prob_size]
    confidence_scores = yolo_output[prob_size: (prob_size + conf_size)]
    cords = yolo_output[(prob_size + conf_size):]

    # Reshape the arrays so that its easier to loop over them
    probabilities = probabilities.reshape((SS, C))
    confs = confidence_scores.reshape((SS, B))
    cords = cords.reshape((SS, B, 4))

    for grid in range(SS):
        for b in range(B):
            bx = Box()

            bx.c = confs[grid, b]

            # bounding box xand y coordinates are offsets of a particular grid cell location,
            # so they are also bounded between 0 and 1.
            # convert them absolute locations relative to the image size
            bx.x = (cords[grid, b, 0] + grid % S) / S
            bx.y = (cords[grid, b, 1] + grid // S) / S


            bx.w = cords[grid, b, 2] ** sqrt
            bx.h = cords[grid, b, 3] ** sqrt

            # multiply confidence scores with class probabilities to get class sepcific confidence scores
            p = probabilities[grid, :] * bx.c

            # Check if the confidence score for class 'car' is greater than the threshold
            if p[car_class_number] >= threshold:
                bx.prob = p[car_class_number]
                boxes.append(bx)

    # combine boxes that are overlap

    # sort the boxes by confidence score, in the descending order
    boxes.sort(key=lambda b: b.prob, reverse=True)


    for i in range(len(boxes)):
        boxi = boxes[i]
        if boxi.prob == 0:
            continue

        for j in range(i + 1, len(boxes)):
            boxj = boxes[j]

            # If boxes have more than 40% overlap then retain the box with the highest confidence score
            if box_iou(boxi, boxj) >= 0.4:
                boxes[j].prob = 0

    boxes = [b for b in boxes if b.prob > 0]

    return boxes


def color_naming(center):
    try:
                actual_color=webcolors.rgb_to_name(tuple(center))
                color_names=str(actual_color)
    except ValueError:
                d={}
                for (key,name_color) in webcolors.css3_hex_to_names.items():
                    r_w,g_w,b_w=webcolors.hex_to_rgb(key)
                    r,g,b=(int(center[0]),int(center[1]),int(center[2]))
                    #calculates hsv values
                    lib_lab=np.uint8(np.asarray([[list((r_w,g_w,b_w))]]))
                    lib_lab=cv2.cvtColor(lib_lab,cv2.COLOR_RGB2Lab)

                    given_lab=np.uint8(np.asarray([[list((r,g,b))]]))
                    given_lab=cv2.cvtColor(given_lab,cv2.COLOR_RGB2Lab)

                    #Extracting individual l,a,b values
                    r_w,g_w,b_w=((lib_lab[0][0])[0],(lib_lab[0][0])[1],(lib_lab[0][0])[2])
                    r,g,b=((given_lab[0][0])[0],(given_lab[0][0])[1],(given_lab[0][0])[2])
                    #r,g,b=colorsys.rgb_to_hsv(center[0]/255.0,center[1]/255.0,center[2]/255.0)
                    #calculate mean squared error  
                    #r=(int(r_w)-int(r))**2
                    g=(int(g_w)-int(g))**2
                    b=(int(b_w)-int(b))**2
                    d[r+g+b]=name_color
                #print(d)
                color_names=d[min(d.keys())]
    return color_names

def centroid_histogram(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
 
	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()
	return hist
    
def draw_boxes(crop_fiename,boxes,im, crop_dim):
    imgcv1 = im.copy()
    
    [xmin, xmax] = crop_dim[0]
    [ymin, ymax] = crop_dim[1]
    
    height, width, _ = imgcv1.shape
    count=1
    for b in boxes:
        w = xmax - xmin
        h = ymax - ymin

        left  = int ((b.x - b.w/2.) * w) + xmin
        right = int ((b.x + b.w/2.) * w) + xmin
        top   = int ((b.y - b.h/2.) * h) + ymin
        bot   = int ((b.y + b.h/2.) * h) + ymin

        if left  < 0:
            left = 0
        if right > width - 1:
            right = width - 1
        if top < 0:
            top = 0
        if bot>height - 1: 
            bot = height - 1
        
        thick = 5 #int((height + width // 150))
        
        cv2.rectangle(imgcv1, (left, top), (right, bot), (255,0,0), thick)
        crop=imgcv1[top:bot,left:right]
        crop_final=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        cv2.imwrite("./output_images/"+crop_filename+"_"+str(count)+".png",crop_final)
        count+=1
        
        
    
##    plt.imshow(cv2.resize(imgcv1, (448,448)))
##    plt.imshow(imgcv1)
##    plt.show()

    return imgcv1

# Load weights
from utils import load_weights

model = get_model()
load_weights(model,'yolo-tiny.weights')
our_images=glob.glob('./test_images/*.jpg')
print(our_images)
for imgs in our_images:
    
    #test_image = mpimg.imread('test_images/test1.png')
    #test_image = plt.imread('test_images/test1.jpg',0)
    filename=imgs
    index=filename.rfind("\\")     
    test_image = plt.imread(filename,0)
    crop_filename=filename[index+1:-4:1]
    pre_processed = preprocess(test_image)
    batch = np.expand_dims(pre_processed, axis=0)
    batch_output = model.predict(batch)
    print(batch_output.shape)

    boxes = yolo_output_to_car_boxes(batch_output[0], threshold=0.25)
    #final = draw_boxes(boxes, test_image, ((500,1280),(300,650)))
    final = draw_boxes(crop_filename,boxes, test_image, ((0,test_image.shape[1]),(0,test_image.shape[0])))


    plt.subplot(1,2,1)
    #plt.imshow(test_image)
    #plt.show()
    plt.axis('off')
    plt.title("Original Image")
    plt.subplot(1,2,2)
    new_final=cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
    cv2.imwrite('output_images/'+crop_filename+"_out"+'.png',new_final)


NUM_CLUSTERS = 5


crop_our_images=glob.glob('./output_images/cropped/*.png')
#print ('reading image')
text_file = open("Output1.txt", "w")
text_file2 = open("Output.txt", "w")
for imgg in crop_our_images:
    fname=imgg
    image=cv2.imread(imgg,1)
    image=cv2.resize(image,(256,256))
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    #cv2.imshow("ima",image)
    cv2.waitKey(5)
    #print(image)
    image=image.reshape((image.shape[0]*image.shape[1],3))
    #print(image)
    clt=KMeans(n_clusters=3)
    clt.fit(image)
    #print(clt.cluster_centers_)

    hist=centroid_histogram(clt)
    dom_color=max(hist)
    lar_ind=np.argmax(hist)
    #print("hello:",hist)
    final_color=clt.cluster_centers_[lar_ind]
    final_color=[int(i) for i in final_color]
    final_color=np.asarray([[np.uint8(np.asarray(final_color))]])
    final_color_hsv=cv2.cvtColor(final_color,cv2.COLOR_RGB2HSV)
    #print(final_color)
    upperbound=[70,70,70]
    if(final_color[0][0][1]>35 and final_color[0][0][2]>40):
            large=np.argmax(hist)
           
    else:
            
            if(hist[0]==dom_color):
                    if(hist[1]>hist[2]):
                            large=1
                    else:
                            large=2
            if(hist[1]==dom_color):
                    if(hist[0]>hist[2]):
                            large=0
                    else:
                            large=2
            if(hist[2]==dom_color):
                    if(hist[0]>hist[1]):
                            large=0
                    else:
                            large=1


    cloe=color_naming(clt.cluster_centers_[large])
    print(cloe)
    
    print(color_naming(clt.cluster_centers_[large]))
    text_file.write(new_filename+" "+color_naming(centers)+"\n")
    text_file2.write((cloe)+" ")
text_file.close()
text_file2.close()
