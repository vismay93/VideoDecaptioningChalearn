from layer import *
from custom_vgg16 import *

class Network:
    def __init__(self, x, y, mask, is_training, batch_size):
        self.batch_size = batch_size
        self.x = tf.concat([x, tf.expand_dims(mask,-1)], axis=-1)     
        #self.imitation,self.mask = self.generator(x , is_training)
        self.imitation = self.generator(self.x , is_training)
        self.mask =  tf.expand_dims(mask,-1)
        self.completion = self.imitation * tf.round(self.mask) + x * (1 - tf.round(self.mask))
        self.perceptual_loss = self.calc_perceptual_loss(self.completion, y)
        self.real = self.discriminator(y, reuse=False)
        self.fake = self.discriminator(self.completion, reuse=True)

        self.g_loss = self.calc_g_loss(y, self.imitation, self.mask)
        self.d_loss = self.calc_d_loss(self.real, self.fake)
        self.gan_loss = self.calc_gan_loss(y,self.imitation, self.mask,self.fake)
        
        self.total_g_loss = tf.add(1e-10*self.perceptual_loss,self.g_loss)
        self.total_gan_loss = tf.add(1e-10*self.perceptual_loss,self.gan_loss)

        self.g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')


    def generator(self, x, is_training):
        with tf.variable_scope('generator'):
            with tf.variable_scope('conv1'):
                x = conv_layer(x, [5, 5, 4, 64], 1)
                x = batch_normalize(x, is_training)
                x_c1 = tf.nn.relu(x)
            with tf.variable_scope('conv2'):
                x = conv_layer(x_c1, [3, 3, 64, 128], 2)
                x = batch_normalize(x, is_training)
                x_c2 = tf.nn.relu(x)
            with tf.variable_scope('conv3'):
                x = conv_layer(x_c2, [3, 3, 128, 128], 1)
                x = batch_normalize(x, is_training)
                x_c3 = tf.nn.relu(x)
            with tf.variable_scope('conv4'):
                x = conv_layer(x_c3, [3, 3, 128, 256], 2)
                x = batch_normalize(x, is_training)
                x_c4 = tf.nn.relu(x)

            with tf.variable_scope('Image'):
                with tf.variable_scope('conv5'):
                    x = conv_layer(x_c4, [3, 3, 256, 256], 1)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv6'):
                    x = conv_layer(x, [3, 3, 256, 256], 1)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('dilated1'):
                    x = dilated_conv_layer(x, [3, 3, 256, 256], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('dilated2'):
                    x = dilated_conv_layer(x, [3, 3, 256, 256], 4)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('dilated3'):
                    x = dilated_conv_layer(x, [3, 3, 256, 256], 8)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('dilated4'):
                    x = dilated_conv_layer(x, [3, 3, 256, 256], 16)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv7'):
                    x = conv_layer(x, [3, 3, 256, 256], 1)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv8'):
                    x = conv_layer(x, [3, 3, 256, 256], 1)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                    x = tf.concat([x, x_c4], axis=-1)
                with tf.variable_scope('deconv1'):
                    x = deconv_layer(x, [4, 4, 128, 512], [self.batch_size, 64, 64, 128], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                    x = tf.concat([x, x_c3], axis=-1)
                with tf.variable_scope('conv9'):
                    x = conv_layer(x, [3, 3, 256, 128], 1)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                    x = tf.concat([x, x_c2], axis=-1)
                with tf.variable_scope('deconv2'):
                    x = deconv_layer(x, [4, 4, 64, 256], [self.batch_size, 128, 128, 64], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                    x = tf.concat([x, x_c1], axis=-1)
                with tf.variable_scope('conv10'):
                    x = conv_layer(x, [3, 3, 128, 32], 1)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv11'):
                    x = conv_layer(x, [3, 3, 32, 3], 1)
                    xi = tf.nn.tanh(x)
        return xi


    def discriminator(self, global_x, reuse):
        def global_discriminator(x):
            is_training = tf.constant(True)
            with tf.variable_scope('global'):
                with tf.variable_scope('conv1'):
                    x = conv_layer(x, [5, 5, 3, 64], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv2'):
                    x = conv_layer(x, [5, 5, 64, 128], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv3'):
                    x = conv_layer(x, [5, 5, 128, 256], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv4'):
                    x = conv_layer(x, [5, 5, 256, 512], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv5'):
                    x = conv_layer(x, [5, 5, 512, 512], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('fc'):
                    x = flatten_layer(x)
                    x = full_connection_layer(x, 1024)
            return x

        with tf.variable_scope('discriminator', reuse=reuse):
            global_output = global_discriminator(global_x)
            with tf.variable_scope('concatenation'):
                output = full_connection_layer(global_output, 1)

        return output

    def calc_g_loss(self, x, completion, mask):
        loss_hole = tf.abs(tf.round(mask) * (x - completion))
        loss_valid = tf.abs(tf.round(1-mask) * (x - completion))
        #loss = tf.nn.l2_loss(tf.round(mask) * (x - completion))
        return tf.add(6*tf.reduce_mean(loss_hole),tf.reduce_mean(loss_valid))

    def calc_gan_loss(self,x,completion, mask, fake):
        alpha = 1e-2
        #fake = self.discriminator(completion, reuse=True)
        loss = tf.add( self.calc_g_loss(x,completion, mask), alpha * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.ones_like(fake))) )
        return loss

    def calc_d_loss(self, real, fake):
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake)))
        return tf.add(d_loss_real, d_loss_fake)

    def calc_perceptual_loss(self, completion, gt):
        data_dict = loadWeightsData('/media/Data1/ashish/anubha/inpainting/glcic/src/vgg16.npy')
        vgg_c = custom_Vgg16(completion, data_dict=data_dict)
        feature_ = [vgg_c.conv2_2, vgg_c.conv3_3, vgg_c.conv4_3, vgg_c.conv5_3]

        vgg = custom_Vgg16(gt, data_dict=data_dict)
        feature = [vgg.conv2_2, vgg.conv3_3, vgg.conv4_3, vgg.conv5_3]

        loss_f = 0
        for f, f_ in zip(feature, feature_):
            lambda_f = 1e0
            loss_f += lambda_f * tf.reduce_mean(tf.subtract(f, f_) ** 2)

        return loss_f
