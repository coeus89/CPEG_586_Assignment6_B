import tensorflow as tf
#import tensorflow.compat.v1 as tf
#from tensorflow import nn.conv2d
import os
from sklearn.utils import shuffle
import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data
#tf.compat.v1.disable_v2_behavior()

class Siamese(object):
    def __init__(self):
        # This is a TensorFlow v1 application so i have to include code to diable some v2 functionality.
        # tf.compat.v1.disable_v2_behavior()
        # tf.compat.v1.disable_eager_execution() # Only needed for v1 code with sessions ect...
        #----set up place holders for inputs and labels for the siamese network---
        # two input placeholders for Siamese network

        self.tf_inputA = tf.placeholder(tf.float32, [None, 784], name = 'inputA')
        self.tf_inputB = tf.placeholder(tf.float32, [None, 784], name = 'inputB')

        # labels for the image pair # 1: similar, 0: dissimilar
        self.tf_Y = tf.placeholder(tf.float32, [None,], name = 'Y')

        # outputs, loss function and training optimizer
        self.outputA, self.outputB = self.siameseNetwork()
        self.loss = self.contrastiveLoss()
        self.optimizer = self.optimizer_initializer()

        # Initialize tensorflow session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def layer(self, tf_input, num_hidden_units, variable_name):
        # tf_input: batch_size x n_features
        # num_hidden_units: number of hidden units
        tf_weight_initializer = tf.random_normal_initializer(mean = 0, stddev = 0.01)
        num_features = tf_input.get_shape()[1] # tf_input.shape[0]????
        W = tf.get_variable(
            name = variable_name + 'W',
            dtype = tf.float32,
            shape=[num_features,num_hidden_units],
            initializer = tf_weight_initializer
        )
        b = tf.get_variable(
            name = variable_name + 'b',
            dtype = tf.float32,
            shape=[num_hidden_units],
        )
        out = tf.add(tf.matmul(tf_input,W), b)
        return out
    
    def CNNLayer(self, tf_input, KernelSize, NumFeatureMaps, variable_name):
        tf_weight_initializer = tf.random_normal_initializer(mean = 0, stdev = 0.01)
        NumFeaturePrevLayer = tf_input.get_shape()[0]
        k = tf.get_variable(
            name = variable_name + "K",
            dtype = tf.float32,
            shape = [NumFeaturePrevLayer,NumFeatureMaps,KernelSize,KernelSize],
            initializer = tf_weight_initializer
        )
        b = tf.get_variable(
            name = variable_name + 'b',
            dtype = tf.float32,
            shape=[NumFeatureMaps]
        )
        # do i need to do a for loop for the different kernels?

        out = tf.add(tf.nn.conv2d(tf_input,k,padding='VALID'),b) # do i need to do strides in conv2d?

    def network(self, tf_input):
        # Setup FNN
        fc1 = self.layer(tf_input = tf_input,num_hidden_units = 1024,variable_name='fc1')
        ac1 = tf.nn.relu(fc1)
        fc2 = self.layer(tf_input = ac1,num_hidden_units=1024,variable_name='fc2')
        ac2 = tf.nn.relu(fc2)
        fc3 = self.layer(tf_input = ac2, num_hidden_units = 2, variable_name = 'fc3')
        return fc3
    
    def siameseNetwork(self):
        with tf.variable_scope("siamese") as scope:
            outputA = self.network(self.tf_inputA)
            # share weights
            scope.reuse_variables()
            outputB = self.network(self.tf_inputB)
        return outputA, outputB
    
    def contrastiveLoss(self,margin = 5.0):
        with tf.variable_scope("siamese") as scope:
            labels = self.tf_Y
            # Euclidean Distance Squared
            dist = tf.pow(tf.subtract(self.outputA,self.outputB),2,name='Dw')
            Dw = tf.reduce_sum(dist,1)
            # add 1e-6 t increase the stability of calculating the gradients
            Dw2 = tf.sqrt(Dw + 1e-6, name = 'Dw2')
            # Loss Function
            lossSimilar =  tf.multiply(labels, tf.pow(Dw2,2), name ='constrastiveLoss_1')
            lossDissimilar = tf.multiply(tf.subtract(1.0, labels),tf.pow(tf.maximum(tf.subtract(margin, Dw2), 0), 2), name = 'constrastiveLoss_2')
            loss = tf.reduce_mean(tf.add(lossSimilar,lossDissimilar), name = 'contrastiveLoss')
            return loss
    
    def optimizer_initializer(self):
        LEARNING_RATE = 0.01
        RAND_SEED = 0 # random seed
        tf.set_random_seed(RAND_SEED)
        # Initialize optimizer
        #optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(self.loss)
        return optimizer

    def trainSiamese(self,mnist,numIterations,batchSize=100):
        # Train the network
        num_batches = mnist[0].shape[0]
        # x_train1,y_train1 = tf.data.Dataset.from_tensor_slices(mnist[0]),tf.data.Dataset.from_tensor_slices(mnist[1])
        # x_train2,y_train2 = tf.data.Dataset.from_tensor_slices(mnist[0]),tf.data.Dataset.from_tensor_slices(mnist[1])
        x_train1,y_train1 = mnist[0],mnist[1]
        x_train2,y_train2 = mnist[0],mnist[1]
        for i in range(numIterations):
            # if (i == 0):
            #     x_train1,y_train1 = shuffle(mnist[0],mnist[1])
            #     x_train2,y_train2 = shuffle(mnist[0],mnist[1])
            iter1 = (i % num_batches) * batchSize
            #input1, y1 = x_train1.batch(batchSize), y_train1.batch(batchSize) 
            input1, y1 = x_train1[iter1:iter1 + batchSize], y_train1[iter1:iter1 + batchSize]
            input2, y2 = x_train2[iter1:iter1 + batchSize], y_train2[iter1:iter1 + batchSize]
            #input1, y1 = mnist.train.next_batch(batchSize)
            #input2, y2 = mnist.train.next_batch(batchSize)
            label = (y1 == y2).astype('float')
            _, trainingLoss = self.sess.run([self.optimizer, self.loss],feed_dict = {self.tf_inputA: input1, self.tf_inputB: input2,self.tf_Y: label})
            if i % 50 == 0:
                print('iteration %d: train loss %.3f' % (i, trainingLoss))

    def test_model(self, input):
        # Test the trained model
        output = self.sess.run(self.outputA, feed_dict = {self.tf_inputA: input})
        return output



