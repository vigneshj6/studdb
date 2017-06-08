
# coding: utf-8

# In[1]:

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import cv2;
from sklearn.linear_model import LogisticRegression;
import random as rand
get_ipython().magic('matplotlib inline')


# In[2]:

from sklearn.decomposition import PCA


# In[3]:

o = pd.read_csv("./Info.txt",delimiter=' ');


# In[4]:

o.head(6)


# In[5]:

print("Tissue : ",o.tissue.unique())
print("Abnormal : ",o.abnormal.unique())
print("Severity : ",o.severity.unique())


# In[6]:

print(o[o.Ref == "mdb005"].severity=='B')


# In[22]:

print (os.getcwd())
images = os.listdir("./input");

train_size = len(images)

APPROX = 10;
count = 0
pixel = 1024;

train_data_X = np.zeros((train_size,pixel,pixel))
train_data_Y = np.zeros((train_size,3),dtype=np.float32)

for i in images:
    img = np.array(cv2.imread('./input/'+i,cv2.CV_8UC1));
    train_data_X[count]=img;
    if np.any(o[o.Ref == i[:-4]].severity=='M'):
        train_data_Y[count][2]=1;
    elif np.any(o[o.Ref == i[:-4]].severity=='B'):
        train_data_Y[count][1]=1;
    else:
        train_data_Y[count][0]=1;
    count = count+1;

print (train_data_X.shape)
print (train_data_Y.shape)
print (train_data_X.dtype)


# In[18]:

"""
#To cut the required part by considering a sum of rows and columns and filtering it by setting a threshold
#column threshold cth
def pre(train_data_X,i):
    cth = 1000;
    #rows threshold rth
    rth = 1000;
    img = train_data_X[i];
    sumrow = np.sum(train_data_X[i], axis=1)
    sumcol = np.sum(train_data_X[i], axis=0)
    print(sumcol.shape)
    k=0
    new_img = [];
    for i in range(1024):
        if(sumcol[i]>cth):
            new_img.append(img[:,i]);
            k=k+1;
    new_img = np.array(new_img);
    new_img2 = [];
    for i in range(1024):
        if(sumrow[i]>rth):
            new_img2.append(new_img[:,i]);
            k=k+1;
    new_img2 = np.array(new_img2,dtype=np.uint8)
    #img = cv2.resize(new_img2,(1024,1024))
    img = new_img2;
    #print(img.max())
    ret, thresh = cv2.threshold(img,20,255,cv2.THRESH_BINARY)
    # You need to choose 4 or 8 for connectivity type
    connectivity = 8
    # Perform the operation
    output = cv2.connectedComponentsWithStats(thresh, connectivity,0)#cv2.CV_32S

    x=0;
    coord = 0;

    for i in output[2][1:]:
        if(i[4]>x):
            x=i[4];
            coord = i;
    for i in output[2][3:]:
        print(i)
        new_img2[i[1]:i[1]+i[3],i[0]:i[2]+i[0]] = 0;

    plt.imshow(new_img2)
    plt.show()
    """


# In[ ]:




# In[51]:

def preprocess(i,name):
    cth = 1000;
    #rows threshold rth
    rth = 1000;
    img = train_data_X[i];
    sumrow = np.sum(train_data_X[i], axis=1)
    sumcol = np.sum(train_data_X[i], axis=0)
    print(sumcol.shape)
    k=0
    new_img = [];
    for i in range(1024):
        if(sumcol[i]>cth):
            new_img.append(img[:,i]);
            k=k+1;
    new_img = np.array(new_img);
    new_img2 = [];
    for i in range(1024):
        if(sumrow[i]>rth):
            new_img2.append(new_img[:,i]);
            k=k+1;
    new_img2 = np.array(new_img2,dtype=np.uint8)
    #img = cv2.resize(new_img2,(1024,1024))
    img = new_img2;
    #print(img.max())
    ret, thresh = cv2.threshold(img,20,255,cv2.THRESH_BINARY)
    # You need to choose 4 or 8 for connectivity type
    connectivity = 8
    # Perform the operation
    output = cv2.connectedComponentsWithStats(thresh, connectivity,0)#cv2.CV_32S

    x=0;
    coord = 0;

    for i in output[2][1:]:
        if(i[4]>x):
            x=i[4];
            coord = i;
    for i in output[2][2:]:
        new_img2[i[1]:i[1]+i[3],i[0]:i[2]+i[0]] = 0;
    image = new_img2
    plt.imsave(fname='output/'+str(name)+'.png',arr=image,cmap='gray')


# In[54]:

images = os.listdir("./input");

train_size = len(images)
pixel = 1024;
for i in range(train_size):
    preprocess(i,images[i])


# In[153]:

print(coord)


# In[99]:

i=25
sumrow = np.sum(train_data_X[i], axis=1)
sumcol = np.sum(train_data_X[i], axis=0)
plt.plot(sumrow)
plt.show()
plt.plot(sumcol)
plt.show()
plt.plot(np.bitwise_or(sumrow>25000,sumcol>2000))
plt.show()
diff = np.minimum(sumrow,sumcol)>500;
plt.plot(diff)
plt.show()
diff = diff.tolist()
start = diff.index(True)
end = start + diff[diff.index(True):].index(False)
plt.imshow(train_data_X[i][start:1000,start:end])
plt.show()
plt.imshow(train_data_X[i])
plt.show()


# In[100]:

def saltandpepper(X,thres):
    Noise = X
    [m,n] = X.shape;
    saltpepper_noise=np.random.rand(m,n); #creates a uniform random variable from 0 to 1 
    for i in range(0,m):
        for j in range(0,n):
            if saltpepper_noise[i,j]<=thres:
                Noise[i,j]=0
            elif saltpepper_noise[i,j]>1-thres:
                Noise[i,j]=255
    return Noise


# In[101]:

def hpf(X):
    print (np.max(X))
    a = np.array([[0,1,0],[1,-4,1],[0,1,0]]);
    return cv2.filter2D(X,-1,a)


# In[154]:

# remove noise
plt.imshow(train_data_X[25])
plt.show()
img = cv2.GaussianBlur(img,(17,17),1)
plt.imshow(img)
plt.show()
# convolute with proper kernels
#laplacian = cv2.Laplacian(img,cv2.CV_8U)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
#plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()


# In[161]:

X = train_data_X[10];
plt.imshow(X,cmap='gray')
plt.show();
laplacian = cv2.Canny(X[10].astype(np.uint8),10,10)
plt.imshow(laplacian)
plt.show();
sobelx = cv2.Sobel(X,cv2.CV_64F,1,0,ksize=5)
plt.imshow(sobelx,cmap='gray')
plt.show();


# In[163]:

X = train_data_X[10];
plt.imshow(X,cmap='gray')
plt.show();
X=saltandpepper(X,0.07)
plt.imshow(X,cmap='gray')
plt.show()
X = cv2.medianBlur(img,3);
plt.imshow(X,cmap='gray')
plt.show()
print(X)
X = hpf(X);
plt.imshow(X,cmap='gray')
plt.show()
print (train_data_Y[9])


# In[168]:

for i in range(30):
    try:
        X = cv2.medianBlur(img,i);
        print(i)
        plt.imshow(X,cmap='gray')
        plt.show()
    except:
        continue;


# In[ ]:




# In[ ]:




# In[ ]:



