#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

import time
from functools import wraps

def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ("Total time running %s: %s seconds" 
                %(function.__name__, str(t1-t0))
                )
        return result
    return function_timer

@fn_timer
def Makedisk(d,center,pixelsize,imagesize,fine):
    halfLenx=pixelsize[0]*(imagesize[0]-1)/2.0
    halfLeny=pixelsize[1]*(imagesize[1]-1)/2.0
    x=np.linspace(-halfLenx,halfLenx,imagesize[0]*fine)
    y=np.linspace(-halfLeny,halfLeny,imagesize[1]*fine)
    X,Y=np.meshgrid(x,y)
    mask= (X-center[0])**2+(Y-center[1])**2<=(d/2.0)**2
    img1=np.zeros(mask.shape,np.float32)
    img1[mask]=1.0

    img=np.zeros(imagesize,np.float32)
    for i in range(imagesize[0]):
        for j in range(imagesize[1]):
            img[i,j]=img1[i*fine:(i+1)*fine,j*fine:(j+1)*fine].mean()
    return img


@fn_timer
def Makedisk0(d,center,pixelsize,imagesize,fine):
    halfLenx=pixelsize[0]*(imagesize[0]-1)/2.0
    halfLeny=pixelsize[1]*(imagesize[1]-1)/2.0
    x=np.linspace(-halfLenx,halfLenx,imagesize[0])
    y=np.linspace(-halfLeny,halfLeny,imagesize[1])
    X,Y=np.meshgrid(x,y)

    plt.figure()
    plt.plot(X,Y,color='red',marker='.',linestyle='')

    rangex=np.array((-imagesize[0]+1,imagesize[0]-1))*pixelsize[0]/2.0
    rangey=np.array((-imagesize[1]+1,imagesize[1]-1))*pixelsize[1]/2.0

    subPixelSize=(pixelsize[0]/fine,pixelsize[1]/fine)
    subBins=(int(d//pixelsize[0]+1)*fine,int(d//pixelsize[1]+1)*fine)
    subXcenter=((center[0]-(rangex[0]-pixelsize[0]/2.0))//pixelsize[0])*pixelsize[0]+rangex[0]
    subYcenter=((center[1]-(rangey[0]-pixelsize[1]/2.0))//pixelsize[1])*pixelsize[1]+rangey[0]
    subCenter=(subXcenter,subYcenter)

    subrangex=(subCenter[0]-(subBins[0]-1)*subPixelSize[0]/2.0,subCenter[0]+(subBins[0]-1)*subPixelSize[0]/2.0)
    subrangey=(subCenter[1]-(subBins[1]-1)*subPixelSize[1]/2.0,subCenter[1]+(subBins[1]-1)*subPixelSize[1]/2.0)
    subx=np.linspace(subrangex[0],subrangex[1],subBins[0])
    suby=np.linspace(subrangey[0],subrangey[1],subBins[1])
    subX,subY=np.meshgrid(subx,suby)
    submask= (subX-center[0])**2+(subY-center[1])**2<=(d/2.0)**2
    img1=np.zeros(submask.shape,np.float32)
    img1[submask]=1.0
    print('img1.shape:'+str(img1.shape))
    print('img1 center:'+str(subCenter))
    print('img1 Bins:'+str(subBins))
    print('img1 subx:'+str(subx))
    print('img1 suby:'+str(suby))

    plt.plot(subX,subY,color='blue',marker='o',linestyle='')
    #plt.plot(subCenter[0],subCenter[1],color='',marker='o',edgecolors='g',s=200)
    plt.plot(subCenter[0],subCenter[1],color='green',marker='o',linestyle='')
    plt.show()

    img=np.zeros(imagesize,np.float32)
    '''
    ind_x_start=int((subx[0]-subPixelSize[0]/2.0-halfLenx)//pixelsize[0])
    ind_y_start=int((suby[0]-subPixelSize[1]/2.0+halfLeny)//pixelsize[1])
    print('x: '+str((ind_x_start,ind_x_end))+ 'y: '+str((ind_y_start,ind_y_end)))
    for i in range(subBins[0]):
        for j in range(subBins[1]):
            img[i,j]=img1[i*fine:(i+1)*fine,j*fine:(j+1)*fine].mean()
    '''
    return img

if __name__ == '__main__':

    d=6.0
    center=(0.0,0.0)
    pixelsize=(2.1,2.1)
    #imagesize=(264,264)
    imagesize=(20,20)
    fine=8
    #img1=Makedisk(d,center,pixelsize,imagesize,fine)
    img2=Makedisk0(d,center,pixelsize,imagesize,fine)

    '''
    plt.figure()
    plt.subplot(131)
    plt.imshow(img1)
    plt.subplot(132)
    plt.imshow(img2)
    plt.subplot(133)
    plt.imshow(img2-img1)
    plt.show()
    '''
