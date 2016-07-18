from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

pickle_file = 'notMNIST.pickle'

num_test_start = 4000
num_test = 1000

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  valid_dataset = valid_dataset[num_test_start:(num_test_start+num_test),:,:]
  valid_labels = valid_labels[num_test_start:(num_test_start+num_test)]
  test_dataset = test_dataset[num_test_start:(num_test_start+num_test),:,:]
  test_labels = test_labels[num_test_start:(num_test_start+num_test)]
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)


image_size = 28
num_labels = 10
num_channels = 1 # grayscale

import numpy as np

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
  #print(predictions)
  #print(labels)
  print(predictions.shape)
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64


graph = tf.Graph()

with graph.as_default():

    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
  
    # Variables.
    def conv2d_s1(x,W):
        return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
    
    def max_pool_2x2(x):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],
                             strides=[1,2,2,1],padding='SAME')
    
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    def bias_variable(shape):
        initial = tf.constant(1.0, shape=shape)
        return tf.Variable(initial)

    W_conv1 = weight_variable([5,5,1,16])
    b_conv1 = bias_variable([16])

    W_conv2 = weight_variable([5,5,16,16])
    b_conv2 = bias_variable([16])

    W_fc1 = weight_variable([7*7*16,64])
    b_fc1 = bias_variable([64])

    W_fc2 = weight_variable([64,10])
    b_fc2 = bias_variable([10])
    
    
    #Model
    def model(data):
        # W_conv1 = weight_variable([5,5,1,16])
        # b_conv1 = bias_variable([16])
    
        h_conv1 = tf.nn.relu(conv2d_s1(data,W_conv1)+b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
        
        # W_conv2 = weight_variable([5,5,16,16])
        # b_conv2 = bias_variable([16])
        
        h_conv2 = tf.nn.relu(conv2d_s1(h_pool1,W_conv2)+b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        
        # W_fc1 = weight_variable([7*7*16,64])
        # b_fc1 = bias_variable([64])
        
        h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*16])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
        
        # W_fc2 = weight_variable([64,10])
        # b_fc2 = bias_variable([10])
        

        return tf.matmul(h_fc1,W_fc2)+b_fc2
  
    # Model.
#     def model(data):
#         conv = conv2d_s1(data, layer1_weights)
#         hidden = tf.nn.relu(conv + layer1_biases)
#         pool = max_pool_2x2(hidden)
#         conv = conv2d_s1(pool,layer2_weights)
#         #conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='SAME')
#         hidden = tf.nn.relu(conv + layer2_biases)
#         pool = max_pool_2x2(hidden)
#         shape = pool.get_shape().as_list()
#         #shape = hidden.get_shape().as_list()
#         reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
#         #reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
#         hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
#         return tf.matmul(hidden, layer4_weights) + layer4_biases
  
    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

num_steps = 3001

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(
                valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))