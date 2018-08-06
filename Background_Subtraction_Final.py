import numpy as np
import cv2 as cv
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import glob
import time
import pickle
import struct
from PIL import Image

import scipy
import scipy.misc
import scipy.cluster
import codecs
import webcolors

NUM_CLUSTERS = 5

color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 15  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True

def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv.cvtColor(image, cv.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv.cvtColor(image, cv.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv.cvtColor(image, cv.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv.cvtColor(image, cv.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

def bin_spatial(img, size=(32, 32)):
    color1 = cv.resize(img[:,:,0], size).ravel()
    color2 = cv.resize(img[:,:,1], size).ravel()
    color3 = cv.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def getmotion(x):
   
    cap = cv.VideoCapture(x)
    
    fgbg = cv.bgsegm.createBackgroundSubtractorGMG()

    l=[]
    count=0
    while(1):
        
        count+=1
        
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        
        #print("hello")
        fmasked = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
        #print("hello2")
        cv.imshow('frame',fmasked)

        if(count%30==0):
            m=[]
            m.append(frame)
            m.append(fmasked)
            l.append(m)
            #print(l)
        k = cv.waitKey(39) & 0xff
        
        if k == 27:
            break
        
    cap.release()
    cv.destroyAllWindows()
    return l
def color_naming(center):
    try:
                actual_color=webcolors.rgb_to_name(tuple(center))
                color_names=str(actual_color)
    except ValueError:
                d={}
                for (key,name_color) in webcolors.css3_hex_to_names.items():
                    r_w,g_w,b_w=webcolors.hex_to_rgb(key)
                    r=(r_w-(center[0]))**2
                    g=(g_w-(center[1]))**2
                    b=(b_w-(center[2]))**2
                    d[r+g+b]=name_color
                color_names=d[min(d.keys())]
    return color_names

def find_dom_color(im):
    ar = np.asarray(im)
    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

    index_max = scipy.argmax(counts)                    # find most frequent
    peak = codes[index_max]
    colour = ''.join(chr(int(c)) for c in peak)
    color=codecs.encode(colour)
    #print(peak,color)
    centers=[]
    i=0
    for i in range(len(peak)):
        centers.append(int(peak[i]))
    return color_naming(centers)                                                        
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7))
e=getmotion("car.mp4")
count1=0
count2=0;
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7))
for i in e:
    count1+=1
    opening = cv.morphologyEx(i[1], cv.MORPH_CLOSE, kernel)
    img=np.copy(i[0])
    ret,thresh = cv.threshold(opening,127,255,0)
    im2, contours, hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, -1, (0,255,0), 3)
    cv.imwrite(str(count1)+str(count2)+'.png',img)
    fincon=[]
    for j in contours:
        area = cv.contourArea(j)
        if(area>200):
            fincon.append(j)
    finconnum=np.asarray(fincon)
   
    for k in finconnum:
        count2+=1
        x,y,w,h = cv.boundingRect(k)
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        crop=img[y:y+h,x:x+w]
        crop64=cv.resize(crop,(64,64))
        cv.imwrite("./crop/"+str(count2)+'.png',crop64)
        
    cv.imwrite(str(count1)+'.png',img)
he=[]
images = glob.glob('./crop/*.png')

    
for image in images:
      
    he.append(image)

car_fea = extract_features(he, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
loaded_model = pickle.load(open("finalized_model.sav", 'rb'))
car_images=[]
lis=loaded_model.predict(car_fea)
for a,b in zip(lis,he):
    if(a==1):
        car_images.append(b)
dom_colors=[]
for img in car_images:
    imgreal=cv.imread(img)
    d=find_dom_color(imgreal)
    dom_colors.append(d)
print(dom_colors)
