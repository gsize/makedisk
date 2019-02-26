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
def Makedisk0(diameter,centerPosition,pixelSize,imageSize,fine):
    '''
    diameter: lesion diameter,unit mm
    centerPosition: lesion center, unit mm
    pixelSize: unit mm
    imageSize: image size
    fine:
    return: image is in Nx_by_Ny
    '''
    Nx=imageSize[0];  Ny=imageSize[1];
    pixelSizeX=pixelSize[0];  pixelSizeY=pixelSize[1];
    centerX=centerPosition[0];  centerY=centerPosition[1];
    
    image=np.zeros((Nx,Ny),np.float32)
    img_cx=Nx*pixelSizeX/2
    img_cy=Ny*pixelSizeY/2
    
    # each pixel is divide into sub-pixels, the weight of poxel is calculated
    #as the sum of sub-pixels (distance <= diameter of lesion) /  sub-pixels
    NumGrid =fine;
    msX= pixelSizeX/NumGrid
    msY = pixelSizeY/NumGrid
    for x in range(Nx):
        px= x*pixelSizeX-img_cx
        for y in range(Ny):
            py =y*pixelSizeY-img_cy
            for x1 in range(NumGrid):
                gx=px+(x1+0.5)*msX
                for y1 in range(NumGrid):
                    gy =py+(y1+0.5)*msY
                    r=( (gy-centerY)**2 + (gx-centerX)**2 )**0.5
                    if r*2<= diameter:
                        image[y,x]+=1.0
    image/=(NumGrid**2)
    #image=np.rot90(image)
    return image


@fn_timer
def Makedisk1(d,center,pixelsize,imagesize,fine):
    rangex=np.array((-imagesize[0]+1,imagesize[0]-1))*pixelsize[0]/2.0
    rangey=np.array((-imagesize[1]+1,imagesize[1]-1))*pixelsize[1]/2.0
    x=np.linspace(rangex[0],rangex[1],imagesize[0]*fine)
    y=np.linspace(rangey[0],rangey[1],imagesize[1]*fine)
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
def Makedisk2(d,center,pixelsize,imagesize,fine):
    rangex=np.array((-imagesize[0]+1,imagesize[0]-1))*pixelsize[0]/2.0
    rangey=np.array((-imagesize[1]+1,imagesize[1]-1))*pixelsize[1]/2.0
    x=np.linspace(rangex[0],rangex[1],imagesize[0])
    y=np.linspace(rangey[0],rangey[1],imagesize[1])
    X,Y=np.meshgrid(x,y)

    subXcenter=((center[0]-(rangex[0]-pixelsize[0]/2.0))//pixelsize[0])*pixelsize[0]+rangex[0]
    subYcenter=((center[1]-(rangey[0]-pixelsize[1]/2.0))//pixelsize[1])*pixelsize[1]+rangey[0]
    subCenter=(subXcenter,subYcenter)
    
    #扩大感兴趣直径，包含圆圈所有点
    vector1=np.array(center)
    vector2=np.array(subCenter)
    dt=np.linalg.norm(vector1-vector2)
    halfbins_x1= int(((d+dt)/pixelsize[0]+1)/2)
    halfbins_y1= int(((d+dt)/pixelsize[1]+1)/2)
    bins_1=(halfbins_x1*2+1,halfbins_y1*2+1)
    range_x1=(subXcenter-halfbins_x1*pixelsize[0],subXcenter+halfbins_x1*pixelsize[0])
    range_y1=(subYcenter-halfbins_y1*pixelsize[1],subYcenter+halfbins_y1*pixelsize[1])
    x1=np.linspace(range_x1[0],range_x1[1],bins_1[0])
    y1=np.linspace(range_y1[0],range_y1[1],bins_1[1])
    X1,Y1=np.meshgrid(x1,y1)
    
    fine=int((fine//2.0)*2+1) #fine取奇数，才可以等分中心点
    
    subPixelSize=(1.0*pixelsize[0]/fine,1.0*pixelsize[1]/fine)
    subHalfBinx=halfbins_x1*fine+fine//2
    subHalfBiny=halfbins_y1*fine+fine//2
    
    subBins=(subHalfBinx*2+1,subHalfBiny*2+1)
    subrangex=(subCenter[0]-subPixelSize[0]*subHalfBinx,subCenter[0]+subPixelSize[0]*subHalfBinx)
    subrangey=(subCenter[1]-subPixelSize[1]*subHalfBiny,subCenter[1]+subPixelSize[1]*subHalfBiny)
    subx=np.linspace(subrangex[0],subrangex[1],subBins[0])
    suby=np.linspace(subrangey[0],subrangey[1],subBins[1])
    subX,subY=np.meshgrid(subx,suby)
    submask= (subX-center[0])**2+(subY-center[1])**2<=(d/2.0)**2
    img1=np.zeros(submask.shape,np.float32)
    img1[submask]=1.0
    
    print('img1 center:'+str(subCenter))
    print('img1 Bins:'+str(subBins))
    print('img1 fine:'+str(fine))
    print('img1 subx:'+str(subx[0])+ ' --  '+str(subx[-1]))
    print('img1 suby:'+str(suby[0])+ ' --  '+str(suby[-1]))

    
    plt.figure()
    plt.plot(X1,Y1,color='red',marker='.',linestyle='')
    plt.plot(subX,subY,color='blue',marker='o',markersize=2,linestyle='',mfc='none')
    plt.plot(subCenter[0],subCenter[1],color='green',marker='o',markersize=8,mfc='none')
    plt.plot(center[0],center[1],color='black',marker='o',markersize=5)
    plt.contour(subX, subY, (subX-center[0])**2 + (subY-center[1])**2, [(d/2.0)**2])
    plt.show()
    
    
    img=np.zeros((imagesize[1],imagesize[0]),np.float32)
    
    ind_x_start=int((subx[0]-rangex[0]+pixelsize[0]/2.0)//pixelsize[0])
    ind_x_end  =int((subx[-1]-rangex[0]+pixelsize[0]/2.0)//pixelsize[0]+1)
    ind_y_start=int((suby[0]-rangey[0]+pixelsize[1]/2.0)//pixelsize[1])
    ind_y_end  =int((suby[-1]-rangey[0]+pixelsize[1]/2.0)//pixelsize[1]+1)
    print('x: '+str((ind_x_start,ind_x_end))+ 'y: '+str((ind_y_start,ind_y_end)))
    for subi,i in zip(range(0,subBins[0],fine), range(ind_x_start,ind_x_end)):
        for subj,j in zip(range(0,subBins[1],fine),range(ind_y_start,ind_y_end) ):
            print('sub: %d,%d   img: %d,%d' %(subi,subj,i,j))
            img[j,i]=img1[subj:subj+fine,subi:subi+fine].mean()
    
    return img

def Plotdiff():
    # figure quality model
    pixelsize=(1.0,2.1)
    imagesize=(264,264)
    fine=4
    
    ballDiameter=np.array([37,17,13,28,22,10])
    centerBallAngle=np.array([40,100,150,220,290,340])*np.pi/180
    centerBallRadius=80/2  # unit mm
    ballPosition =(np.cos(centerBallAngle)*centerBallRadius,np.sin(centerBallAngle)*centerBallRadius)
    
    center=(ballPosition[0][0],ballPosition[1][0])
    print('center position: '+str(center))
    img0=Makedisk0(ballDiameter[0],center,pixelsize,imagesize,fine)
    img2=Makedisk2(ballDiameter[0],center,pixelsize,imagesize,fine)
    
    
    plt.figure()
    plt.subplot(131)
    plt.imshow(img0)
    plt.subplot(132)
    plt.imshow(img2)
    plt.subplot(133)
    plt.imshow(img2-img0)
    

def InitLesion_par():
    # figure quality model
    pixelsize=(2.1,2.1)
    imagesize=(264,264)
    fine=3
    
    ballDiameter=np.array([37,17,13,28,22,10])
    centerBallAngle=np.array([40,100,150,220,290,340])*np.pi/180
    centerBallRadius=80/2  # unit mm
    ballPosition =(np.cos(centerBallAngle)*centerBallRadius,np.sin(centerBallAngle)*centerBallRadius)
    
    #plt.figure()
    #fig,axs= plt.subplots(nrows=1,ncols=len(thresholds),figsize=(20,5));
    img0List=[]
    img1List=[]
    img2List=[]
    for i in range(len(ballDiameter)):
    #for i in range(1):
        center=(ballPosition[0][i],ballPosition[1][i])
        img0=Makedisk0(ballDiameter[i],center,pixelsize,imagesize,fine)
        img1=Makedisk1(ballDiameter[i],center,pixelsize,imagesize,fine)
        img2=Makedisk2(ballDiameter[i],center,pixelsize,imagesize,fine)
        img0List.append(img0)
        img1List.append(img1)
        img2List.append(img2)
    img0=np.array(img0List).sum(axis=0)
    img1=np.array(img1List).sum(axis=0)
    img2=np.array(img2List).sum(axis=0)
    plt.figure()
    plt.subplot(231)
    plt.imshow(img0)
    plt.subplot(232)
    plt.imshow(img1)
    plt.subplot(233)
    plt.imshow(img2)
    plt.subplot(234)
    plt.imshow(img1-img0)
    plt.subplot(235)
    plt.imshow(img2-img0)


if __name__ == '__main__':

    #InitLesion_par()
    Plotdiff()
    #imagesize=(20,20)
    
    #'''
    x=np.linspace(-100,100,200)
    y=np.linspace(-60,60,120)
    
    X,Y=np.meshgrid(x,y)
    center=(30,40)
    d=16
    mask=(X+center[0])**2+(Y+center[1])**2<(d/2)**2
    mask[100,50]=1
    plt.figure()
    plt.subplot(121)
    plt.plot(X,Y,color='gray',marker='.',markersize=2,linestyle='')
    plt.contour(X, Y, (X+center[0])**2 + (Y+center[1])**2, [(d/2)**2])
    plt.subplot(122)
    plt.imshow(mask)
