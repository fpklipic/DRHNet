import tensorflow as tf
import numpy as np

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

def GroupNorm(x,G=32,eps=1e-5):
    N,H,W,C=x.shape        
    x=tf.reshape(x,[tf.cast(N,tf.int32),tf.cast(H,tf.int32),tf.cast(W,tf.int32),tf.cast(G,tf.int32),tf.cast(C//G,tf.int32)])
    mean,var=tf.nn.moments(x,[1,2,4],keep_dims=True)
    x=(x-mean)/tf.sqrt(var+eps)
    x=tf.reshape(x,[tf.cast(N,tf.int32),tf.cast(H,tf.int32),tf.cast(W,tf.int32),tf.cast(C,tf.int32)])
    gamma = tf.Variable(tf.ones(shape=[1,1,1,tf.cast(C,tf.int32)]), name="gamma")
    beta = tf.Variable(tf.zeros(shape=[1,1,1,tf.cast(C,tf.int32)]), name="beta")
    return x*gamma+beta

def weight_variable(shape):
    initial = tf.truncated_normal(shape, mean=0.0,stddev=np.sqrt(2.0/shape[2]))
    weight_decay = 0.004
    new_weight_decay = tf.multiply(tf.nn.l2_loss(initial), weight_decay)
    tf.add_to_collection('losses', new_weight_decay)
    return tf.Variable( initial)

def identity_block(X_input, kernel_size, in_filter, out_filters, stage, block, training=True):
        """
        Implementation of the identity block as defined in Figure 3

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        training -- train or test

        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        block_name = 'res' + str(stage) + block
        f1, f2, f3 = out_filters
        with tf.variable_scope(block_name):
            X_shortcut = X_input
            #first
            W_conv1 = weight_variable([1, 1, in_filter, f1])
            X = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1], padding='VALID')
            X = GroupNorm(X)
            X = OurRelu(X)
            #second
            W_conv2 = weight_variable([kernel_size, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
            X = GroupNorm(X)
            X = OurRelu(X)
            #third
            W_conv3 = weight_variable([1, 1, f2, f3])
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
            X = GroupNorm(X)
            #final step
            add = tf.add(X, X_shortcut)
            add_result = OurRelu(add)
        return add_result

def convolutional_block(X_input, kernel_size, in_filter,
                            out_filters, stage, block, training=True, stride=2):
        """
        Implementation of the convolutional block as defined in Figure 4

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        training -- train or test
        stride -- Integer, specifying the stride to be used

        Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        block_name = 'res' + str(stage) + block
        with tf.variable_scope(block_name):
            f1, f2, f3 = out_filters

            x_shortcut = X_input
            #first
            W_conv1 = weight_variable([1, 1, in_filter, f1])
            X = tf.nn.conv2d(X_input, W_conv1,strides=[1, stride, stride, 1],padding='VALID')
            X = GroupNorm(X)
            X = LeakyRelu(X)
            #second
            W_conv2 = weight_variable([kernel_size, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1,1,1,1], padding='SAME')
            X = GroupNorm(X)
            X = LeakyRelu(X)
            #third
            W_conv3 = weight_variable([1,1, f2,f3])
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1,1], padding='VALID')
            X = GroupNorm(X)
            W_shortcut = weight_variable([1, 1, in_filter, f3])
            x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, stride, stride, 1], padding='SAME')
            #final
            add = tf.add(x_shortcut, X)
            add_result = LeakyRelu(add)
        return add_result
