#author : ANUBHA PANDEY
import numpy as np
import tensorflow as tf
import cv2
import tqdm
from network_modified import Network
import load
from data_loader import *
import scipy.misc

IMAGE_SIZE = 128
HOLE_MIN = 24
HOLE_MAX = 48
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
NO_FRAMES = 2
PRETRAIN_EPOCH = 1
kernel1 = np.ones((5,5),np.uint8)
kernel2 = np.ones((3,3),np.uint8)

DATASET_PATH = './Data'
#'/media/Data2/vismay/anubha/frames'

with tf.device('/device:GPU:0'):
	#dl = dataLoader('/media/Data1/ashish/anubha/inpainting/glcic/data/images','train',NO_FRAMES,BATCH_SIZE,IMAGE_SIZE)
	dl = dataLoader(DATASET_PATH,'train',NO_FRAMES,BATCH_SIZE,IMAGE_SIZE)
	N = dl.nofiles()
	step_num = int(N / BATCH_SIZE)
	idx = [i for i in range(N)]

	np.random.shuffle(idx)
	for i in tqdm.tqdm(range(step_num)):
		x_batch,y_batch = dl.getbatchFrame(idx[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
		#x_batch_g = 0.299*x_batch[:,:,:,0] + 0.587*x_batch[:,:,:,1] + 0.114*x_batch[:,:,:,2]
		#y_batch_g = 0.299*y_batch[:,:,:,0] + 0.587*y_batch[:,:,:,1] + 0.114*y_batch[:,:,:,2]

		#f = np.mean(x_batch,axis=3)-np.mean(y_batch,axis=3)
		f = x_batch - y_batch
		f = np.abs(f)
		#f=np.sum(f, axis = 3)
		f[f<=0.21]=0
		f[f>0.21]=1
		f = np.sum(f,axis=3)
		f[f>0] = 1
		#f = cv2.morphologyEx(f,cv2.MORPH_OPEN,kernel)
		for j in range(len(f)):
			#p = cv2.morphologyEx(f[j],cv2.MORPH_OPEN,kernel1)
			p = cv2.dilate(f[j],kernel2,iterations = 2)
			#p = cv2.morphologyEx(p, cv2.MORPH_OPEN, kernel1)
			#p = cv2.morphologyEx(p, cv2.MORPH_CLOSE, kernel1)
			s = DATASET_PATH+'./mask/mask'+str(j)+'.jpg'
			scipy.misc.imsave(s,p*255)
			"""
			s = './mask/input'+str(j)+'.jpg'
			scipy.misc.imsave(s,x_batch[j])
			s = './mask/output'+str(j)+'.jpg'
			scipy.misc.imsave(s,y_batch[j])
			"""

		print(np.shape(f))
		exit()
