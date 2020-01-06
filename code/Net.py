import tensorflow as tf
import numpy as np
import resnet_block

def LeakyRelu(x, leak=0.2, name="LeakyRelu"):
    with tf.variable_scope(name):
        leak_c = tf.constant(0.1)
        leak = tf.Variable(leak_c)
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)
    
def OurRelu(x, name="OurRelu"):
    with tf.variable_scope(name):
        leak_c = tf.constant(0.1)
        leak = tf.Variable(leak_c)
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * tf.abs(x) - f2 * x
    
def Friend_relu(x):
    x = tf.nn.relu(x)
    Max = tf.constant([255.0])
    return tf.minimum(x, Max)
    
#normalization
def Batch_normalization(X):
    _mean, _var = tf.nn.moments(X, [0, 1, 2])
    X = tf.nn.batch_normalization(X, _mean, _var, 0, 1, 0.0001)
    return X

#group normalization
def GroupNorm(x,G=32,eps=1e-5):
    N,H,W,C=x.shape     
    x=tf.reshape(x,[tf.cast(N,tf.int32),tf.cast(H,tf.int32),tf.cast(W,tf.int32),tf.cast(G,tf.int32),tf.cast(C//G,tf.int32)])
#     x=tf.reshape(x,[N,H,W,G,C//G])
    mean,var=tf.nn.moments(x,[1,2,4],keep_dims=True)
    x=(x-mean)/tf.sqrt(var+eps)
    x=tf.reshape(x,[tf.cast(N,tf.int32),tf.cast(H,tf.int32),tf.cast(W,tf.int32),tf.cast(C,tf.int32)])
    gamma = tf.Variable(tf.ones(shape=[1,1,1,tf.cast(C,tf.int32)]), name="gamma")
    beta = tf.Variable(tf.zeros(shape=[1,1,1,tf.cast(C,tf.int32)]), name="beta")
    return x*gamma+beta


class Net:
    def __init__(self):
        pass
    
    #kernel initial
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, mean=0.0,stddev=np.sqrt(2.0/shape[2]))
        return tf.Variable( initial)
    
    #bias initial
    def bias_variable(self,shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.1))
    
    def model(self, input_X, training):
        
        #Multi-scale Convolution
        w_conv1_3 = self.weight_variable([3, 3, 3, 64])
        x_conv1_3 = tf.nn.conv2d(input_X, w_conv1_3, strides=[1, 2, 2, 1], padding='SAME')#64 x 64 x64
        w_conv1_5 = self.weight_variable([5, 5, 3, 32])
        x_conv1_5 = tf.nn.conv2d(input_X, w_conv1_5, strides=[1, 2, 2, 1], padding='SAME')
        w_conv1_7 = self.weight_variable([7, 7, 3, 32])
        x_conv1_7 = tf.nn.conv2d(input_X, w_conv1_7, strides=[1, 2, 2, 1], padding='SAME')
        x_conv1 = tf.concat([x_conv1_3, x_conv1_5, x_conv1_7],3)
        x_conv1 = GroupNorm(x_conv1)
        x_conv1 = LeakyRelu(x_conv1)
        w_conv2 = self.weight_variable([3, 3, 128, 256])
        x_conv2 = tf.nn.conv2d(x_conv1, w_conv2, strides=[1, 2, 2, 1], padding='SAME')#32 x32 x128
        x_conv2 = GroupNorm(x_conv2)
        x_conv2 = LeakyRelu(x_conv2)
        w_conv4 = self.weight_variable([3, 3, 256, 512])
        x_conv4 = tf.nn.conv2d(x_conv2, w_conv4, strides=[1, 2, 2, 1], padding='SAME')#16x16x256
        x_conv4 = GroupNorm(x_conv4)
        x_conv4 = LeakyRelu(x_conv4)
        x_conv6 = resnet_block.identity_block(x_conv4, 3, 512, [256, 256, 512], stage=2, block='b', training=training )
        x_conv7 = resnet_block.identity_block(x_conv6, 3, 512, [256, 256, 512], stage=2, block='c', training=training )
        x_conv8 = resnet_block.identity_block(x_conv7, 3, 512, [256, 256, 512], stage=2, block='d', training=training )
        x_conv8 = resnet_block.identity_block(x_conv8, 3, 512, [256, 256, 512], stage=2, block='e', training=training )
        x_conv8 = resnet_block.identity_block(x_conv8, 3, 512, [256, 256, 512], stage=2, block='f', training=training )
        x_conv8 = resnet_block.identity_block(x_conv8, 3, 512, [256, 256, 512], stage=2, block='g', training=training )
        x_conv8 = resnet_block.identity_block(x_conv8, 3, 512, [256, 256, 512], stage=2, block='h', training=training )
        w_deconv1 = self.weight_variable([1, 1, 512, 512])
        x_conv9 = tf.nn.conv2d_transpose(x_conv8, w_deconv1,output_shape=tf.shape(x_conv4), strides=[1, 1, 1, 1], padding='VALID')#29x29x256
        x_conv9 = GroupNorm(x_conv9)
        x_conv9 = OurRelu(x_conv9)
        x_conv9 = tf.concat([x_conv9, x_conv4],3)
        w_conv9_1 = self.weight_variable([1, 1, 1024, 512])
        x_conv9 = tf.nn.conv2d(x_conv9, w_conv9_1, strides=[1, 1, 1, 1], padding='VALID')
        x_conv9 = GroupNorm(x_conv9)
        x_conv9 = LeakyRelu(x_conv9)
        w_deconv2 = self.weight_variable([3, 3, 256, 512])
        x_conv10 = tf.nn.conv2d_transpose(x_conv9, w_deconv2,output_shape=tf.shape(x_conv2), strides=[1, 2, 2, 1], padding='SAME')
        x_conv10 = GroupNorm(x_conv10)
        x_conv10 = OurRelu(x_conv10)
        x_conv10 = tf.concat([x_conv10, x_conv2],3)
        w_conv10_1 = self.weight_variable([1, 1, 512, 256])
        x_conv10 = tf.nn.conv2d(x_conv10, w_conv10_1, strides=[1, 1, 1, 1], padding='SAME')
        x_conv10 = GroupNorm(x_conv10)
        x_conv10 = LeakyRelu(x_conv10)
        w_deconv3 = self.weight_variable([3, 3, 128, 256])
        x_conv11 = tf.nn.conv2d_transpose(x_conv10, w_deconv3,output_shape=tf.shape(x_conv1), strides=[1, 2, 2, 1], padding='SAME')
        x_conv11 = GroupNorm(x_conv11)
        x_conv11 = OurRelu(x_conv11)
        x_conv11 = tf.concat([x_conv11, x_conv1],3)
        w_conv11_1 = self.weight_variable([1, 1, 256, 128])
        x_conv11 = tf.nn.conv2d(x_conv11, w_conv11_1, strides=[1, 1, 1, 1], padding='VALID')
        x_conv11 = GroupNorm(x_conv11)
        x_conv11 = LeakyRelu(x_conv11)
        w_deconv4 = self.weight_variable([3, 3, 3, 128])
        x_conv12 = tf.nn.conv2d_transpose(x_conv11, w_deconv4,output_shape=tf.shape(input_X), strides=[1, 2, 2, 1], padding='SAME')
        model = tf.add(x_conv12,input_X)
        model = Friend_relu(model)
        return input_X,x_conv12,model
    
if __name__ == "__main__":
    net = Net()
    input_X = tf.placeholder(tf.float32, [None, 128,128,3])
    model = net.model(input_X,training=True)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    pre = sess.run(model)
    print(pre.shape)