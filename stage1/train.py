import numpy as np
import tensorflow as tf
import cv2
import tqdm
from network import Network
from data_loader import *

IMAGE_SIZE = 128
HOLE_MIN = 24
HOLE_MAX = 48
LEARNING_RATE = 1e-4
BATCH_SIZE = 6
NO_FRAMES = 2
PRETRAIN_EPOCH = 2
kernel = np.ones((3,3),np.uint8)

#'../scratch/backup_mask4'
#'/media/Data2/vismay/anubha/frames'
PRETRAINED_PATH = '../model/stage1'
DATASET_PATH = '../Data/frames/train'


def train():
	with tf.device('/device:GPU:0'):
		x = tf.placeholder(tf.float32, [BATCH_SIZE*NO_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3])
		y = tf.placeholder(tf.float32, [BATCH_SIZE*NO_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3])
		maskin = tf.placeholder(tf.float32, [BATCH_SIZE*NO_FRAMES, IMAGE_SIZE, IMAGE_SIZE])
		is_training = tf.placeholder(tf.bool, [])

		model = Network(x, y, maskin, is_training, batch_size=BATCH_SIZE*NO_FRAMES)
		#sess = tf.Session()
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
		global_step = tf.Variable(0, name='global_step', trainable=False)
		epoch = tf.Variable(0, name='epoch', trainable=False)

		opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
		g_train_op = opt.minimize(model.g_loss, global_step=global_step, var_list=model.g_variables)
		d_train_op = opt.minimize(model.d_loss, global_step=global_step, var_list=model.d_variables)
		gan_train_op = opt.minimize(model.gan_loss, global_step=global_step, var_list=model.g_variables)


		#train_writer = tf.summary.FileWriter('./backup_mask' + '/train')
		init_op = tf.global_variables_initializer()
		sess.run(init_op)

		if tf.train.get_checkpoint_state(PRETRAINED_PATH):
			saver = tf.train.Saver()
			saver.restore(sess, PRETRAINED_PATH+'/1latest')

		#find total no videos
		#dl = dataLoader('/media/Data1/ashish/anubha/inpainting/glcic/data/images','train',NO_FRAMES,BATCH_SIZE,IMAGE_SIZE)
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
					x_batch,y_batch = dl.getbatchFrame(idx[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
					f = x_batch - y_batch
					f = np.abs(f)
					f[f<=0.21]=0
					f[f>0.21]=1
					f = np.sum(f,axis=3)
					f[f>0] = 1
					for j in range(len(f)):
						#f[j] = cv2.morphologyEx(f[j],cv2.MORPH_OPEN,kernel)
						f[j] = cv2.dilate(f[j],kernel,iterations = 2)

					_, g_loss, m_loss = sess.run([g_train_op, model.g_loss, model.mask_loss], feed_dict={x: x_batch, y: y_batch, maskin: f,is_training: True})
					#tf.summary.scalar('generator_loss', g_loss)
					"""
					if(i%500 == 0):
						comp,mask = sess.run([model.completion,model.mask],feed_dict={x: x_batch, y: y_batch, maskin: f, is_training: True})
						save_images('../scratch/backup_mask4/train',sess.run(epoch),i,x_batch,comp,mask,f)
						#collect_image_summaries(x_batch,model.completion,model.completion)
					"""
					print('Completion loss: {}'.format(g_loss))
					print('Mask loss: {}'.format(m_loss))

					g_loss_value += g_loss

				print('Completion loss for Epoch: {}'.format(g_loss_value))

				saver = tf.train.Saver()
				saver.save(sess, PRETRAINED_PATH+'/' + str(sess.run(epoch)) + 'latest', write_meta_graph=False)
				if sess.run(epoch) == PRETRAIN_EPOCH:
					saver.save(sess, PRETRAINED_PATH+'/pretrained', write_meta_graph=False)


			# Discrimitation
			else:
				gan_loss_value = 0
				d_loss_value = 0
				for i in tqdm.tqdm(range(step_num)):
					x_batch,y_batch = dl.getbatchFrame(idx[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
					f = x_batch - y_batch
					f = np.abs(f)
					f[f<=0.21]=0
					f[f>0.21]=1
					f = np.sum(f,axis=3)
					f[f>0] = 1
					for j in range(len(f)):
						#f[j] = cv2.morphologyEx(f[j],cv2.MORPH_OPEN,kernel)
						f[j] = cv2.dilate(f[j],kernel,iterations = 2)

					_, gan_loss, m_loss, completion = sess.run([gan_train_op, model.gan_loss, model.mask_loss, model.completion], feed_dict={x: x_batch, y: y_batch, maskin: f, is_training: True})
					gan_loss_value += gan_loss
					_, d_loss = sess.run([d_train_op, model.d_loss], feed_dict={x: x_batch, y: y_batch, maskin: f, is_training: True})
					d_loss_value += d_loss
					#tf.summary.scalar('generator_loss', g_loss)
					#tf.summary.scalar('descriminator_loss', d_loss)
					"""
					if(i%500 == 0):
						comp,mask = sess.run([model.completion,model.mask],feed_dict={x: x_batch, y: y_batch, maskin: f, is_training: True})
						save_images('../scratch/backup_mask4/train',sess.run(epoch),i,x_batch,comp,mask,f)
					"""
					print('Completion loss: {}'.format(gan_loss))
					print('Mask loss: {}'.format(m_loss))
					print('Discriminator loss: {}'.format(d_loss))
					

				print('Completion loss for Epoch: {}'.format(gan_loss_value))
				print('Discriminator loss for Epoch: {}'.format(d_loss_value))
				saver = tf.train.Saver()
				saver.save(sess, PRETRAINED_PATH+'/' + str(sess.run(epoch)) + 'latest', write_meta_graph=False)

def collect_image_summaries(gt,completion,mask):
	tf.summary.image('gt_image', gt)
	tf.summary.image('generated_image', completion)
	tf.summary.image('mask',mask)

def save_images(pre,ep,id,gt,comp,mask,maskin):
	for i,g in enumerate(gt):
		cv2.imwrite(pre+'/epoch'+str(ep)+'_iter'+str(id)+'_bid'+str(i)+'gt.jpg',(g[...,[2,1,0]]+1)*127.5)
		cv2.imwrite(pre+'/epoch'+str(ep)+'_iter'+str(id)+'_bid'+str(i)+'gen.jpg',(comp[i][...,[2,1,0]]+1)*127.5)
		cv2.imwrite(pre+'/epoch'+str(ep)+'_iter'+str(id)+'_bid'+str(i)+'mask.jpg',mask[i]*127.5)
		cv2.imwrite(pre+'/epoch'+str(ep)+'_iter'+str(id)+'_bid'+str(i)+'maskgt.jpg',maskin[i]*127.5)

if __name__ == '__main__':
 train()
