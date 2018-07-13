import numpy as np
import tensorflow as tf
import cv2
import tqdm
import os
import matplotlib.pyplot as plt
import sys
from network_perceptual_skipconn import Network
from data_loader_stage2 import *


IMAGE_SIZE = 128
HOLE_MIN = 24
HOLE_MAX = 48
BATCH_SIZE = 1
NO_FRAMES = 5
PRETRAIN_EPOCH = 1
TOTAL_FRAMES = 125

#savepath = '../../scratch/output/val_6/'
#test_npy = './lfw.npy'
#PRETRAINED_PATH = '../../scratch/backup_perceptual_skipconn/15latest'
savepath = '../Output/stage2/'
DATASET_PATH = '../Output/stage1/'
PRETRAINED_PATH = '../model/stage2/'
part = 'val'

def test():
    with tf.device('/device:GPU:0'):
        x = tf.placeholder(tf.float32, [BATCH_SIZE*NO_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3])
        y = tf.placeholder(tf.float32, [BATCH_SIZE*NO_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3])
        maskin = tf.placeholder(tf.float32, [BATCH_SIZE*NO_FRAMES, IMAGE_SIZE, IMAGE_SIZE])
        is_training = tf.placeholder(tf.bool, [])

        model = Network(x, y, maskin, is_training, batch_size=BATCH_SIZE*NO_FRAMES)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        saver = tf.train.Saver()
        saver.restore(sess, PRETRAINED_PATH + '15latest')
        print("model loaded")
        
        dl = dataLoader(DATASET_PATH,part,NO_FRAMES,BATCH_SIZE,IMAGE_SIZE, False)        
         
        N = dl.nofiles()
        #N = 0
        print(N)

        fstep = int(TOTAL_FRAMES / NO_FRAMES)
        writer = tf.summary.FileWriter('logs', sess.graph)
        writer.close()
        count = 0
        for i in tqdm.tqdm(range(N)):
            x_batch,m_batch = dl.getTestbatch(i) 
            Y = []
            for j in range(fstep):
                x_clip = x_batch[j*NO_FRAMES:(j+1)*NO_FRAMES]
                m_clip = m_batch[j*NO_FRAMES:(j+1)*NO_FRAMES]
                
                completion = sess.run(model.completion, feed_dict={x: x_clip, maskin: m_clip, is_training: False})
                Y.append(completion)
                
            Y = np.asarray(Y)
            Y = Y.reshape((Y.shape[0]*Y.shape[1], Y.shape[2], Y.shape[3], Y.shape[4]))
            dl.saveVideo(savepath+'Y', 'Y', count, Y)

            count = count+1

if __name__ == '__main__':
    test()
