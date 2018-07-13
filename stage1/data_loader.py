#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from PIL import Image
import numpy as np
import pims
import subprocess as sp
import re
import os, sys
import pickle
import cv2
import scipy.misc as sm
import shutil
from os import path

class dataLoader:
    def __init__(self,basepath,part,no_frames,batchsize,fsize=128,isTrain=True,isVid=True):
        self.basepath = basepath
        self.part = part
        self.no_frames = no_frames
        self.batchsize = batchsize
        self.fsize = fsize
        self.isTrain = isTrain        
        if isTrain:
            if isVid:
                self.filelistX,self.filelistY = self.getallfiles()
            else:
                self.filelistX,self.filelistY = self.getalldir()
        else:
            self.filelistX = self.getXfiles()

        self.novid = self.nofiles()

    def __len__(self):
        return len(self.filelistX)*self.no_frames

    def nofiles(self):
        return len(self.filelistX)

    def getallfiles(self):
        fX = []
        d = self.basepath + '/' + self.part + '/X/'
        for root, _, fnames in sorted(os.walk(d)):
            fX.extend(fnames)

        fY = []
        d = self.basepath + '/' + self.part + '/Y/'
        for root, _, fnames in sorted(os.walk(d)):
            fY.extend(fnames)

        return sorted(fX),sorted(fY)

    def getalldir(self):
        d = self.basepath + '/' + self.part + '/X/'
        fX = os.listdir(d)

        d = self.basepath + '/' + self.part + '/Y/'
        fY = os.listdir(d)

        return sorted(fX),sorted(fY)


    def getXfiles(self):
        fX = []
        d = self.basepath + '/' + self.part + '/X/'
        for root, _, fnames in sorted(os.walk(d)):
            fX.extend(fnames)

        return sorted(fX)

    def getTrainfiles(self):
        #load training files for stage 2 
        fM = []
        d = self.basepath + '/train_output/mask/' 
        for root, _, fnames in sorted(os.walk(d)):
            fM.extend(fnames)

        fX = []
        d = self.basepath + '/train_output/X/' 
        for root, _, fnames in sorted(os.walk(d)):
            fX.extend(fnames)

        fY = []
        d = self.basepath+'/frames/'+self.part+'/Y/'
        for root, _, fnames in sorted(os.walk(d)):
            fY.extend(fnames)        

        return sorted(fX),sorted(fY),sorted(fM)

    def getbatch(self,idx):
        X = []
        Y = []
        #Read a batch of clips from files
        for i in idx:
            idfs = list(range(25*5))
            np.random.shuffle(idfs)
            idfs = idfs[:self.no_frames] # keep only 2 random frames per clip on train mode
            ok = True
            try:
                #print(self.basepath + '/' + self.part + '/X/'+self.filelistX[i])
                Xj = pims.Video(self.basepath + '/' + self.part + '/X/'+self.filelistX[i])[idfs]
                Xj = np.array(Xj, dtype='float32') / 255.
                #print(self.basepath + '/' + self.part + '/Y/'+self.filelistY[i])
                Yj = pims.Video(self.basepath + '/' + self.part + '/Y/'+self.filelistY[i])[idfs]
                Yj = np.array(Yj, dtype='float32') / 255.

            except:
                print('Error clip number '+ str(i) + ' at  '+ self.filelistX[i] + ' OR '+ self.filelistY[i])
                ok = False
            if ok:
                X.append(Xj)
                Y.append(Yj)

        # make numpy and reshape
        X = np.asarray(X)
        X = X.reshape((X.shape[0]*X.shape[1], X.shape[2], X.shape[3], X.shape[4]))
        Y = np.asarray(Y)
        Y = Y.reshape((Y.shape[0]*Y.shape[1], Y.shape[2], Y.shape[3], Y.shape[4]))
        return X*2 - 1, Y*2 - 1

    def getTestbatch(self,i):
        X = []
        #Read a batch of clips from files
        #for i in idx:
        idfs = list(range(25*5))
        #np.random.shuffle(idfs)
        #idfs = idfs[:self.no_frames] # keep only 2 random frames per clip on train mode
        ok = True
        try:
            #print(self.basepath + '/' + self.part + '/X/'+self.filelistX[i])
            Xj = pims.Video(self.basepath + '/' + self.part + '/X/'+self.filelistX[i])[idfs]
            Xj = np.array(Xj, dtype='float32') / 255
        except:
            print('Error clip number '+ str(i) + ' at  '+ self.filelistX[i])
            ok = False
        if ok:
            X.append(Xj)

        # make numpy and reshape
        X = np.asarray(X)
        print(np.shape(X))
        X = X.reshape((X.shape[0]*X.shape[1], X.shape[2], X.shape[3], X.shape[4]))
        return X*2 - 1
        #return X


    def getbatchFrame(self,idx):
        X = []
        Y = []
        #Read a batch of clips from files
        for i in idx:
            idfs = list(range(25*5))
            np.random.shuffle(idfs)
            idfs = idfs[:self.no_frames] # keep only 2 random frames per clip on train mode
            ok = True
            try:
                Xj = []
                for fid in idfs:
                    im = cv2.imread(self.basepath+'/'+self.part+'/X/'+self.filelistX[i]+'/'+str(fid)+'.jpg')
                    Xj.append(im[...,[2,1,0]])
                    #im = sm.imread(self.basepath+'/'+self.part+'/X/'+self.filelistX[i]+'/'+str(fid)+'.jpg')
                    #Xj.append(im[...,[2,1,0]])
                Xj = np.array(Xj, dtype='float32') / 255.

                Yj = []
                for fid in idfs:
                    im = cv2.imread(self.basepath+'/'+self.part+'/Y/'+self.filelistY[i]+'/'+str(fid)+'.jpg')
                    Yj.append(im[...,[2,1,0]])
                    #im = sm.imread(self.basepath+'/'+self.part+'/Y/'+self.filelistY[i]+'/'+str(fid)+'.jpg')
                    #Yj.append(im[...,[2,1,0]])
                Yj = np.array(Yj, dtype='float32') / 255.

            except:
                print('Error clip number '+ str(i) + ' at  '+ self.filelistX[i] + ' OR '+ self.filelistY[i])
                ok = False
            if ok:
                X.append(Xj)
                Y.append(Yj)

        # make numpy and reshape
        X = np.asarray(X)
        X = X.reshape((X.shape[0]*X.shape[1], X.shape[2], X.shape[3], X.shape[4]))
        Y = np.asarray(Y)
        Y = Y.reshape((Y.shape[0]*Y.shape[1], Y.shape[2], Y.shape[3], Y.shape[4]))
        return X*2-1, Y*2-1

    def getTrainbatchFrame(self,idx):
        #load training data frames for stage2
        X = []
        Y = []
        M = []
        #Read a batch of clips from files
        for i in idx:
            idfs = list(range(25*5))
            np.random.shuffle(idfs)
            idfs = idfs[:self.no_frames] # keep only 2 random frames per clip on train mode
            ok = True
            filelistMask
            try:
                Xj = []
                for fid in idfs:
                    im = cv2.imread(self.basepath+'/train_output/X/'+self.filelistX[i]+'/'+str(fid)+'.jpg')
                    Xj.append(im[...,[2,1,0]])
                    #im = sm.imread(self.basepath+'/'+self.part+'/X/'+self.filelistX[i]+'/'+str(fid)+'.jpg')
                    #Xj.append(im[...,[2,1,0]])
                Xj = np.array(Xj, dtype='float32') / 255.

                Yj = []
                for fid in idfs:
                    im = cv2.imread(self.basepath+'/frames/'+self.part+'/Y/'+self.filelistY[i]+'/'+str(fid)+'.jpg')
                    Yj.append(im[...,[2,1,0]])
                    #im = sm.imread(self.basepath+'/'+self.part+'/Y/'+self.filelistY[i]+'/'+str(fid)+'.jpg')
                    #Yj.append(im[...,[2,1,0]])
                Yj = np.array(Yj, dtype='float32') / 255.
                
                Mj = []
                for fid in idfs:
                    im = cv2.imread(self.basepath+'/train_output/mask/mask'+self.filelistX[i][2:]+'/'+str(fid)+'.jpg')
                    Mj.append(im[...,[2,1,0]])
                Mj = np.array(Mj, dtype='float32') / 255.

            except:
                print('Error clip number '+ str(i) + ' at  '+ self.filelistX[i] + ' OR '+ self.filelistY[i])
                ok = False
            if ok:
                X.append(Xj)
                Y.append(Yj)
                M.append(Mj)
        # make numpy and reshape
        X = np.asarray(X)
        X = X.reshape((X.shape[0]*X.shape[1], X.shape[2], X.shape[3], X.shape[4]))
        Y = np.asarray(Y)
        Y = Y.reshape((Y.shape[0]*Y.shape[1], Y.shape[2], Y.shape[3], Y.shape[4]))
        M = np.asarray(M)
        M = M.reshape((M.shape[0]*M.shape[1], M.shape[2], M.shape[3], M.shape[4]))        
        return X*2-1, Y*2-1, M*2-1        

    def getbatchFrame3d(self,idx):
        X = []
        Xc = []
        Y = []
        #Read a batch of clips from files
        for i in idx:
            idfs = list(range(25*5-2))
            np.random.shuffle(idfs)
            idfs = idfs[:self.no_frames] # keep only 2 random frames per clip on train mode
            ok = True
            try:
                Xj = []
                Xcj = []
                for fid in idfs:
                    im1 = cv2.imread(self.basepath+'/'+self.part+'/X/'+self.filelistX[i]+'/'+str(fid)+'.jpg')
                    im2 = cv2.imread(self.basepath+'/'+self.part+'/X/'+self.filelistX[i]+'/'+str(fid+1)+'.jpg')
                    im3 = cv2.imread(self.basepath+'/'+self.part+'/X/'+self.filelistX[i]+'/'+str(fid+2)+'.jpg')
                    imf = np.concatenate((im1[...,[2,1,0]],im2[...,[2,1,0]],im3[...,[2,1,0]]),axis=2)
                    Xcj.append(imf)
                    Xj.append(im2[...,[2,1,0]])
                Xj = np.array(Xj, dtype='float32') / 255.
                Xcj = np.array(Xcj, dtype='float32') / 255.

                Yj = []
                for fid in idfs:
                    im = cv2.imread(self.basepath+'/'+self.part+'/Y/'+self.filelistY[i]+'/'+str(fid+1)+'.jpg')
                    Yj.append(im[...,[2,1,0]])
                Yj = np.array(Yj, dtype='float32') / 255.

            except:
                print('Error clip number '+ str(i) + ' at  '+ self.filelistX[i] + ' OR '+ self.filelistY[i])
                ok = False
            if ok:
                X.append(Xj)
                Xc.append(Xcj)
                Y.append(Yj)


        # make numpy and reshape
        X = np.asarray(X)
        X = X.reshape((X.shape[0]*X.shape[1], X.shape[2], X.shape[3], X.shape[4]))
        Xc = np.asarray(Xc)
        Xc = Xc.reshape((Xc.shape[0]*Xc.shape[1], Xc.shape[2], Xc.shape[3], Xc.shape[4]))
        Y = np.asarray(Y)
        Y = Y.reshape((Y.shape[0]*Y.shape[1], Y.shape[2], Y.shape[3], Y.shape[4]))
        return X*2-1, Xc*2-1, Y*2-1

    def getbatchFrame5d(self,idx):
        X = []
        Xc = []
        Y = []
        #Read a batch of clips from files
        for i in idx:
            idfs = list(range(25*5-4))
            np.random.shuffle(idfs)
            idfs = idfs[:self.no_frames] # keep only 2 random frames per clip on train mode
            ok = True
            try:
                Xj = []
                Xcj = []
                for fid in idfs:
                    im1 = cv2.imread(self.basepath+'/'+self.part+'/X/'+self.filelistX[i]+'/'+str(fid)+'.jpg')
                    im2 = cv2.imread(self.basepath+'/'+self.part+'/X/'+self.filelistX[i]+'/'+str(fid+1)+'.jpg')
                    im3 = cv2.imread(self.basepath+'/'+self.part+'/X/'+self.filelistX[i]+'/'+str(fid+2)+'.jpg')
                    im4 = cv2.imread(self.basepath+'/'+self.part+'/X/'+self.filelistX[i]+'/'+str(fid+3)+'.jpg')
                    im5 = cv2.imread(self.basepath+'/'+self.part+'/X/'+self.filelistX[i]+'/'+str(fid+4)+'.jpg')
                    imf = np.concatenate((im1[...,[2,1,0]],im2[...,[2,1,0]],im3[...,[2,1,0]],im4[...,[2,1,0]],im5[...,[2,1,0]]),axis=2)
                    Xcj.append(imf)
                    Xj.append(im2[...,[2,1,0]])
                Xj = np.array(Xj, dtype='float32') / 255.
                Xcj = np.array(Xcj, dtype='float32') / 255.

                Yj = []
                for fid in idfs:
                    im = cv2.imread(self.basepath+'/'+self.part+'/Y/'+self.filelistY[i]+'/'+str(fid+1)+'.jpg')
                    Yj.append(im[...,[2,1,0]])
                Yj = np.array(Yj, dtype='float32') / 255.

            except:
                print('Error clip number '+ str(i) + ' at  '+ self.filelistX[i] + ' OR '+ self.filelistY[i])
                ok = False
            if ok:
                X.append(Xj)
                Xc.append(Xcj)
                Y.append(Yj)


        # make numpy and reshape
        X = np.asarray(X)
        X = X.reshape((X.shape[0]*X.shape[1], X.shape[2], X.shape[3], X.shape[4]))
        Xc = np.asarray(Xc)
        Xc = Xc.reshape((Xc.shape[0]*Xc.shape[1], Xc.shape[2], Xc.shape[3], Xc.shape[4]))
        Y = np.asarray(Y)
        Y = Y.reshape((Y.shape[0]*Y.shape[1], Y.shape[2], Y.shape[3], Y.shape[4]))
        return X*2-1, Xc*2-1, Y*2-1        


    def saveFrames(self,writebase):
        for i in range(self.novid):
            ok = True
            try:
                Xj = pims.Video(self.basepath + '/' + self.part + '/X/'+self.filelistX[i])
                if(self.isTrain):
                    Yj = pims.Video(self.basepath + '/' + self.part + '/Y/'+self.filelistY[i])
            except:
                print('Error clip number '+ str(i) + ' at  '+ self.filelistX[i] + ' OR '+ self.filelistY[i])
                ok = False
            if ok:
                wp = writebase + '/' + self.part + '/X/' + self.filelistX[i][:-4]
                if not os.path.exists(wp):
                    os.makedirs(wp)
                for j in range(len(Xj)):
                    cv2.imwrite(wp + '/' + str(j) + '.jpg',Xj[j][...,[2,1,0]])

                if self.isTrain:
                    wp = writebase + '/' + self.part + '/Y/' + self.filelistY[i][:-4]
                    if not os.path.exists(wp):
                        os.makedirs(wp)
                    for j in range(len(Yj)):
                        cv2.imwrite(wp + '/' + str(j) + '.jpg',Yj[j][...,[2,1,0]])

    def saveValFrames(self,writebase):
        d = self.basepath + '/' + self.part + '/X/'
        fX = os.listdir(d)
        for i in range(len(fX)):
            d = d + fX[i] + '/'
            fx = []
            for root, _, fnames in sorted(os.walk(d)):
                fx.extend(fnames)
            for j in range(len(fx)):
                Xj = pims.Video(d+fx[j])
                wp = writebase + '/' + dl.part + '/X/' + fX[i]
                if not os.path.exists(wp):
                    os.makedirs(wp)
                for j in range(len(Xj)):
                    cv2.imwrite(wp + '/' + str(j) + '.jpg',Xj[j][...,[2,1,0]])

    def saveVideo(self, savepath, name, i, clip):
        clip = (clip + 1) * 127.5
        
        
        #clip = clip * 255
        clip = clip.astype('uint8')
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        # write video stream #
        command = ['ffmpeg',
        '-y',  # overwrite output file if it exists
        '-f', 'rawvideo',
        '-s', '128x128', #'256x256', # size of one frame
        '-pix_fmt', 'rgb24',
        '-r', '25', # frames per second
        '-an',  # Tells FFMPEG not to expect any audio
        '-i', '-',  # The input comes from a pipe
        '-vcodec', 'libx264',
        '-b:v', '100k',
        '-vframes', '125', # 5*25
        '-s', '128x128', #'256x256', # size of one frame
        savepath+'/'+name+str(i)+'.mp4'] #savepath+'/Y'+self.filelistX[i][1:]]

        pipe = sp.Popen( command, stdin=sp.PIPE, stderr=sp.PIPE)
        out, err = pipe.communicate(clip.tostring())
        pipe.wait()
        pipe.terminate()
        #print(err)




'''
fsize = 128 # 256

#root_dataset = '../dataset-mp4/' # Download from competition url
#root_dataset = '../dataset-sample'
#root_dataset = '../../data/dataset-mp4/'
root_dataset = '/media/data/Datasets/Inpainting'
'''
'''
data generator used for baseline1
load video clip and randomly choose 2 frames for training
'''
'''
def get_batch(idx, batchsize, max_samples, no_frame, part): #part = train|dev|test
    i = 64 * (idx-1)
    X = []
    Y = []
    #Read a batch of clips from files
    j = 0
    while len(X) < batchsize:
        print (len(X))
        idxs = list(range(25*5))
        np.random.shuffle(idxs)
        idxs = idxs[:no_frame] # keep only 2 random frames per clip on train mode
        print(idxs)
        ok = True
        try:
            Xj = pims.Video(root_dataset+'/'+part+'/X/X'+str(i+j)+'.mp4')[idxs]
            Xj = np.array(Xj, dtype='float32') / 255.
            Yj = pims.Video(root_dataset+'/'+part+'/Y/Y'+str(i+j)+'.mp4')[idxs]
            Yj = np.array(Yj, dtype='float32') / 255.
        except:
            print('Error clip number '+ str(i+j) + ' at  '+root_dataset+'/'+part+'/X/X'+str(i+j)+'.mp4'+ ' OR '+root_dataset+'/'+part+'/Y/Y'+str(i+j)+'.mp4')
            ok = False
            if i+j >= max_samples: j = 0
        if ok:
            X.append(Xj)
            Y.append(Yj)
        j = j + 1

    # make numpy and reshape
    X = np.asarray(X)
    X = X.reshape((X.shape[0]*X.shape[1], X.shape[2], X.shape[3], X.shape[4]))
    Y = np.asarray(Y)
    Y = Y.reshape((Y.shape[0]*Y.shape[1], Y.shape[2], Y.shape[3], Y.shape[4]))
    return (X, Y)

def generate_data(max_samples, batchsize, part): #part = train|dev|test
    while 1:
        samples = list(range(0, max_samples, batchsize))
        #np.random.shuffle(samples)
        for i in samples:
            X = []
            Y = []

            #Read a batch of clips from files
            j = 0
            while len(X) < batchsize:
                if part == 'train':
                    idxs = list(range(25*5))
                    np.random.shuffle(idxs)
                    idxs = idxs[:2] # keep only 2 random frames per clip on train mode
                else:
                    idxs = [50, 100] # only evaluate frames 50 and 100 on eval mode

                ok = True
                try:
                    Xj = pims.Video(root_dataset+'/'+part+'/X/X'+str(i+j)+'.mp4')[idxs]
                    Xj = np.array(Xj, dtype='float32') / 255.
                    Yj = pims.Video(root_dataset+'/'+part+'/Y/Y'+str(i+j)+'.mp4')[idxs]
                    Yj = np.array(Yj, dtype='float32') / 255.
                except:
                    print('Error clip number '+ str(i+j) + ' at  '+root_dataset+'/train/X/X'+str(i+j)+'.mp4'+ ' OR '+root_dataset+'/train/Y/Y'+str(i+j)+'.mp4')
                    ok = False
                    if i+j >= max_samples: j = 0
                if ok:
                    X.append(Xj)
                    Y.append(Yj)
                j = j + 1

            # make numpy and reshape
            X = np.asarray(X)
            X = X.reshape((X.shape[0]*X.shape[1], X.shape[2], X.shape[3], X.shape[4]))
            Y = np.asarray(Y)
            Y = Y.reshape((Y.shape[0]*Y.shape[1], Y.shape[2], Y.shape[3], Y.shape[4]))
            yield (X, Y)

# return all frames from video clip
# returned frames are normalized
def getAllFrames(clipname):
    print(clipname)

    # open one video clip sample
    try:
        data = pims.Video(root_dataset+'/'+clipname)
    except:
        data = pims.Video(clipname)

    data = np.array(data, dtype='float32')
    length = data.shape[0]

    return data[:125] / 255.

# create video clip using 'ffmpeg' command
# clip: input data, supposed normalized (between 0 and 1)
# name: basename of output file
def createVideoClip(clip, folder, name):
    clip = clip * 255.
    clip = clip.astype('uint8')

    # write video stream #
    command = [ 'ffmpeg',
    '-y',  # overwrite output file if it exists
    '-f', 'rawvideo',
    '-s', '128x128', #'256x256', # size of one frame
    '-pix_fmt', 'rgb24',
    '-r', '25', # frames per second
    '-an',  # Tells FFMPEG not to expect any audio
    '-i', '-',  # The input comes from a pipe
    '-vcodec', 'libx264',
    '-b:v', '100k',
    '-vframes', '125', # 5*25
    '-s', '128x128', #'256x256', # size of one frame
    folder+'/'+name+'.mp4' ]

    pipe = sp.Popen( command, stdin=sp.PIPE, stderr=sp.PIPE)
    out, err = pipe.communicate(clip.tostring())
    pipe.wait()
    pipe.terminate()
    #print(err)

################################### baseline2 #################################

# for baseline2, we precompute mini batches.
# don't need a generator since inputs are small dimension (patches)
def all_files(d):
    f = []
    for root, _, fnames in sorted(os.walk(d)):
        f.extend(fnames)
    return f

def build_and_save_batches(batchsize): #part = train|dev|test
        different_clips_per_batch = 10
        number_of_frames_per_clips = 2
        fx = all_files(root_dataset+'/train/X')
        fy = all_files(root_dataset+'/train/Y')

        max_samples = len(fx)
        print(max_samples)
        samples = list(range(max_samples))
        np.random.shuffle(samples)
        num_batch = 0
        for i in range(0, max_samples, different_clips_per_batch):
            X = []
            Y = []

            #Read a batch of clips from files
            j = 0
            while len(X) < different_clips_per_batch:
                idxs = list(range(25*5))
                np.random.shuffle(idxs)
                idxs = idxs[:number_of_frames_per_clips] # keep only 2 random frames per clip
                print(i)
                print(j)
                print('read clip '+str(samples[i+j])+' at idxs '+str(idxs))
                ok = True

                try:
                    Xj = pims.Video(root_dataset+'/train/X/'+(fx[samples[i+j]]))[idxs]
                    Xj = np.array(Xj, dtype='float32') / 255.

                    Yj = pims.Video(root_dataset+'/train/Y/'+(fy[samples[i+j]]))[idxs]
                    Yj = np.array(Yj, dtype='float32') / 255.
                except:
                    print('Error clip number '+ fx[samples[i+j]] + ' at '+root_dataset+'/train/X/'+fx[samples[i+j]]+ ' OR '+root_dataset+'/train/Y/'+fy[samples[i+j]])
                    ok = False
                if ok:
                    X.append(Xj)
                    Y.append(Yj)
                j = j + 1

            # get random non-overlapped patches
            X = np.asarray(X)
            X = X.reshape((X.shape[0]*X.shape[1], X.shape[2], X.shape[3], X.shape[4]))
            X = X.reshape(-1, fsize//32,32,fsize//32,32, 3).swapaxes(2,3).reshape(-1,32,32,3)

            Y = np.asarray(Y)
            Y = Y.reshape((Y.shape[0]*Y.shape[1], Y.shape[2], Y.shape[3], Y.shape[4]))
            Y = Y.reshape(-1, fsize//32,32,fsize//32,32, 3).swapaxes(2,3).reshape(-1,32,32,3)

            # compute differnce to look for patches including text
            # wrong image comparison, should use opencv or PILLOW, but ok..
            Tt = abs(X - Y)
            T=np.array([np.max(t) for t in Tt])
            T[T>0.2] = 1
            T[T<0.2] = 0

            # get random positive and negative patches
            Tpos_idxs = np.where(T>0)[0]
            np.random.shuffle(Tpos_idxs)
            Tneg_idxs = np.where(T==0)[0]
            np.random.shuffle(Tneg_idxs)

            # try to make nbpos = nbneg = batchsize/2
            nbpos = int(batchsize/2)
            if len(Tpos_idxs) < nbpos: nbpos = len(Tpos_idxs)

            # shuffle idxs
            patch_idxs = np.concatenate([Tpos_idxs[:nbpos], Tneg_idxs[:int(batchsize-nbpos)]])
            np.random.shuffle(patch_idxs)
            X = X[patch_idxs]
            Y = Y[patch_idxs]
            T = T[patch_idxs]

            # save in pickle
            data = (X,Y,T)
            with open('batches/batch_'+str(num_batch)+'.pkl', 'wb') as f:
                print('write batch '+str(num_batch))
                pickle.dump(data, f)
                num_batch = num_batch + 1


# load and return minibatches for training
def load_batches(idxfrom, idxto): # 0, 3500
    train_batches = []
    for i in range(idxfrom, idxto):
        with open('batches/batch_'+str(i)+'.pkl', 'rb') as f:
            train_batches.append(pickle.load(f))
    return train_batches

if __name__ == "__main__":
    if sys.argv[1] == 'build_and_save_batches': build_and_save_batches(10)
'''
