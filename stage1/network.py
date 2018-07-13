from layer import *

class Network:
    def __init__(self, x, y, maskin, is_training, batch_size):
        self.batch_size = batch_size
        self.imitation,self.mask = self.generator(x , is_training)
        self.completion = self.imitation #* tf.round(self.mask) + x * (1 - tf.round(self.mask))

        self.real = self.discriminator(y, reuse=False)
        self.fake = self.discriminator(self.completion, reuse=True)
        self.mask_loss = self.calc_mask_loss(self.mask[:,:,:,0],maskin)
        self.g_loss = self.calc_g_loss(y, self.completion,self.mask[:,:,:,0],maskin)
        self.d_loss = self.calc_d_loss(self.real, self.fake)
        self.gan_loss = self.calc_gan_loss(y,self.completion,self.mask[:,:,:,0],maskin)
        self.g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')


    def generator(self, x, is_training):
        with tf.variable_scope('generator'):
            with tf.variable_scope('conv1'):
                x = conv_layer(x, [5, 5, 3, 64], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv2'):
                x = conv_layer(x, [3, 3, 64, 128], 2)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv3'):
                x = conv_layer(x, [3, 3, 128, 128], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv4'):
                x = conv_layer(x, [3, 3, 128, 256], 2)
                x = batch_normalize(x, is_training)
                xb = tf.nn.relu(x)

            with tf.variable_scope('Image'):
                with tf.variable_scope('conv5'):
                    x = conv_layer(xb, [3, 3, 256, 256], 1)
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
                with tf.variable_scope('deconv1'):
                    x = deconv_layer(x, [4, 4, 128, 256], [self.batch_size, 64, 64, 128], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv9'):
                    x = conv_layer(x, [3, 3, 128, 128], 1)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('deconv2'):
                    x = deconv_layer(x, [4, 4, 64, 128], [self.batch_size, 128, 128, 64], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv10'):
                    x = conv_layer(x, [3, 3, 64, 32], 1)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv11'):
                    x = conv_layer(x, [3, 3, 32, 3], 1)
                    xi = tf.nn.tanh(x)
            with tf.variable_scope('Mask'):
                with tf.variable_scope('conv5'):
                    x = conv_layer(xb, [3, 3, 256, 256], 1)
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
                with tf.variable_scope('deconv1'):
                    x = deconv_layer(x, [4, 4, 128, 256], [self.batch_size, 64, 64, 128], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv9'):
                    x = conv_layer(x, [3, 3, 128, 128], 1)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('deconv2'):
                    x = deconv_layer(x, [4, 4, 64, 128], [self.batch_size, 128, 128, 64], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv10'):
                    x = conv_layer(x, [3, 3, 64, 32], 1)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv11'):
                    x = conv_layer(x, [3, 3, 32, 1], 1)
                    xm = tf.nn.sigmoid(x)

        return xi,xm


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
            output = global_discriminator(global_x)
            #with tf.variable_scope('concatenation'):
            #    output = full_connection_layer(global_output, 1)

        return output


    def calc_mask_loss(self,mask,maskin):
        loss = tf.nn.l2_loss(mask - maskin)
        return tf.reduce_mean(loss)

    def calc_g_loss(self, x, completion, mask, maskin):
        loss = tf.nn.l2_loss(x - completion)
        beta = 1e-6
        return tf.add( tf.reduce_mean(loss), beta * self.calc_mask_loss(mask,maskin) )

    def calc_gan_loss(self,x,completion,mask,maskin):
        alpha = 1e-2
        fake = self.discriminator(completion, reuse=True)
        loss = tf.add( self.calc_g_loss(x,completion,mask,maskin), alpha * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.ones_like(fake))) )
        return loss

    def calc_d_loss(self, real, fake):
        #alpha = 4e-4
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake)))
        return tf.add(d_loss_real, d_loss_fake)