import numpy as np
import tensorflow as tf
import cv2
import tqdm
from network_perceptual_skipconn import Network
import load
from data_loader_stage2 import *
import scipy.misc

IMAGE_SIZE = 128
HOLE_MIN = 24
HOLE_MAX = 48
LEARNING_RATE = 1e-3
BATCH_SIZE = 12
NO_FRAMES = 1
PRETRAIN_EPOCH = 2

PRETRAIN_PATH = '../model/stage2/'
DATASET_PATH = '../Output/stage1'

def train():
	with tf.device('/device:GPU:0'):
	    x = tf.placeholder(tf.float32, [BATCH_SIZE*NO_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3])
	    y = tf.placeholder(tf.float32, [BATCH_SIZE*NO_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3])
	    mask = tf.placeholder(tf.float32, [BATCH_SIZE*NO_FRAMES, IMAGE_SIZE, IMAGE_SIZE])
	    is_training = tf.placeholder(tf.bool, [])

	    model = Network(x, y, mask,is_training, batch_size=BATCH_SIZE*NO_FRAMES)
	    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
	    global_step = tf.Variable(0, name='global_step', trainable=False)
	    epoch = tf.Variable(0, name='epoch', trainable=False)

	    opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
	    g_train_op = opt.minimize(model.total_g_loss, global_step=global_step, var_list=model.g_variables)
	    d_train_op = opt.minimize(model.d_loss, global_step=global_step, var_list=model.d_variables)
	    gan_train_op = opt.minimize(model.total_gan_loss, global_step=global_step, var_list=model.g_variables)

	    init_op = tf.global_variables_initializer()
	    sess.run(init_op)

	    if tf.train.get_checkpoint_state(PRETRAIN_PATH):
	        saver = tf.train.Saver()
	        saver.restore(sess, PRETRAIN_PATH+'1latest')

	    #find total no videos
	    dl = dataLoader(DATASET_PATH,'train',NO_FRAMES,BATCH_SIZE,IMAGE_SIZE)
	    N = dl.nofiles()
	    step_num = int(N / BATCH_SIZE)
	    idx = [i for i in range(N)]

	    while True:
	        sess.run(tf.assign(epoch, tf.add(epoch, 1)))
	        print('epoch: {}'.format(sess.run(epoch)))

	        np.random.shuffle(idx)
	        # Completion
	        if sess.run(epoch) <= PRETRAIN_EPOCH:
	            g_loss_value = 0
	            for i in tqdm.tqdm(range(step_num)):
	                x_batch,y_batch,m_batch = dl.getTrainbatchFrame(idx[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
	                _, total_g_loss,p_loss = sess.run([g_train_op, model.total_g_loss, model.perceptual_loss], feed_dict={x: x_batch, y: y_batch, mask: m_batch, is_training: True})
	                print('G loss: {}'.format(total_g_loss))
	                print('P loss: {}'.format(p_loss))
	                g_loss_value += total_g_loss

	            print('G loss for Epoch: {}'.format(g_loss_value))
	            saver = tf.train.Saver()
	            saver.save(sess, PRETRAIN_PATH + str(sess.run(epoch)) + 'latest', write_meta_graph=True)
	            if sess.run(epoch) == PRETRAIN_EPOCH:
	                saver.save(sess, PRETRAIN_PATH +'pretrained', write_meta_graph=False)
	        # Discrimitation
	        else:
	            gan_loss_value = 0
	            d_loss_value = 0
	            for i in tqdm.tqdm(range(step_num)):
	                x_batch,y_batch,m_batch = dl.getTrainbatchFrame(idx[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
	                _, total_gan_loss, g_loss, p_loss = sess.run([gan_train_op, model.total_gan_loss, model.g_loss, model.perceptual_loss], feed_dict={x: x_batch, y: y_batch, mask:m_batch, is_training: True})
	                gan_loss_value += total_gan_loss
	                _, d_loss = sess.run([d_train_op, model.d_loss], feed_dict={x: x_batch, y: y_batch, mask: m_batch, is_training: True})
	                d_loss_value += d_loss
	                print('GAN loss: {}'.format(total_gan_loss))
	                print('G loss: {}'.format(g_loss))
	                print('P loss: {}'.format(p_loss))
	                print('Discriminator loss: {}'.format(d_loss))

	            print('GAN loss for Epoch: {}'.format(gan_loss_value))
	            print('Discriminator loss for Epoch: {}'.format(d_loss_value))
	            saver = tf.train.Saver()
	            saver.save(sess, PRETRAIN_PATH + str(sess.run(epoch)) + 'latest', write_meta_graph=True)

def save_images(pre,ep,id,gt,comp,mask,inp):
	for i,g in enumerate(gt):
	    cv2.imwrite(pre+'/epoch'+str(ep)+'_iter'+str(id)+'_bid'+str(i)+'gt.jpg',(g+1)*255./2.)
	    cv2.imwrite(pre+'/epoch'+str(ep)+'_iter'+str(id)+'_bid'+str(i)+'gen.jpg',(comp[i]+1)*255./2.)
	    cv2.imwrite(pre+'/epoch'+str(ep)+'_iter'+str(id)+'_bid'+str(i)+'mask.jpg',(mask[i]+1)*255/2.)
	    cv2.imwrite(pre+'/epoch'+str(ep)+'_iter'+str(id)+'_bid'+str(i)+'inp.jpg',(inp[i]+1)*255./2.)


if __name__ == '__main__':
    train()
