import numpy as np
import tensorflow as tf
import cv2
import tqdm
import os
import matplotlib.pyplot as plt
import sys
from network import Network
from data_loader import *


IMAGE_SIZE = 128
HOLE_MIN = 24
HOLE_MAX = 48
BATCH_SIZE = 1
NO_FRAMES = 5
PRETRAIN_EPOCH = 1
TOTAL_FRAMES = 125

#PRETRAIN_PATH = '../../scratch/backup_mask4/10latest'
#savepath = '/media/Data2/vismay/anubha/train'
#DATASET_PATH = '/media/Data1/ashish/anubha/inpainting/glcic/data/images/'
#savepath = '../../scratch/output/val_4/'
#savepath = '/media/Data1/ashish/anubha/inpainting/glcic/src/test/train_output/'
#'/media/Data2/vismay/anubha/train_output_stage1/'
#test_npy = './lfw.npy'


PRETRAIN_PATH= '../model/stage1/10latest'
savepath = '../Output/stage1'
DATASET_PATH= '../Data/videos/train'
part = 'train'

def test():
    with tf.device('/device:GPU:1'):
        x = tf.placeholder(tf.float32, [BATCH_SIZE*NO_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3])
        y = tf.placeholder(tf.float32, [BATCH_SIZE*NO_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3])
        maskin = tf.placeholder(tf.float32, [BATCH_SIZE*NO_FRAMES, IMAGE_SIZE, IMAGE_SIZE])
        is_training = tf.placeholder(tf.bool, [])

        model = Network(x, y, maskin, is_training, batch_size=BATCH_SIZE*NO_FRAMES)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        saver = tf.train.Saver()
        saver.restore(sess, PRETRAIN_PATH )
        print("model loaded")
        dl = dataLoader(DATASET_PATH,part,NO_FRAMES,BATCH_SIZE,IMAGE_SIZE,False)
         
        N = dl.nofiles()
        print(N)
        
        fstep = int(TOTAL_FRAMES / NO_FRAMES)
        writer = tf.summary.FileWriter('logs', sess.graph)
        writer.close()
        count = 0
        for i in tqdm.tqdm(range(N)):
            x_batch = dl.getTestbatch(i) 

            Y = []
            I = []
            M = []
            for j in range(fstep):
                x_frame = x_batch[j*NO_FRAMES:(j+1)*NO_FRAMES] 
                imitation,mask = sess.run([model.imitation,model.mask], feed_dict={x: x_frame, is_training: False})
                maskb = mask > 0.1
                maskb = np.concatenate((maskb, maskb, maskb), axis=-1)
                completion = imitation * maskb + x_frame * (1 - maskb)
                Y.append(completion)
                I.append(imitation)
                mask = np.concatenate((mask, mask, mask), axis=-1)
                M.append(mask)

            Y = np.asarray(Y)
            Y = Y.reshape((Y.shape[0]*Y.shape[1], Y.shape[2], Y.shape[3], Y.shape[4]))

            M = np.asarray(M)
            M = M.reshape((M.shape[0]*M.shape[1], M.shape[2], M.shape[3], M.shape[4]))

            I = np.asarray(I)
            I = I.reshape((I.shape[0]*I.shape[1], I.shape[2], I.shape[3], I.shape[4]))

            count = count+1
            id=str(dl.filelistX[i].split('.')[0].split('X')[1])
            save_images(savepath,id,Y,M)

def save_images(pre,id,Y,mask):                                                
    wp_x = pre+'/X'+'/X'+str(id)
    wp_mask = pre+'/mask'+'/mask'+str(id)
    if not os.path.exists(wp_x):
        os.makedirs(wp_x)
    if not os.path.exists(wp_mask):
        os.makedirs(wp_mask)       
    for i,g in enumerate(Y):         
        cv2.imwrite(wp_x+'/'+str(i)+'.jpg',(g+1)*255./2.)
        cv2.imwrite(wp_mask+'/'+str(i)+'.jpg',(mask[i]+1)*255./2.)


if __name__ == '__main__':
    test()
