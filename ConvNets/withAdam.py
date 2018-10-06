import tensorflow as tf
import numpy as np 
import os
import pickle

data_root = "/Users/amajha/Documents/ml_data"
pickle_file = 'notMNIST.pickle'
image_size = 28
num_classes = 10
num_channels = 1 

with open(os.path.join(data_root, pickle_file), 'rb') as pk:
    dataset = pickle.load(pk)
    train_dataset = dataset['train_dataset']
    train_labels = dataset ['train_labels']
    test_dataset = dataset['test_dataset']
    test_labels = dataset['test_labels']
    valid_dataset = dataset['valid_dataset']
    valid_labels = dataset['valid_labels']

    del dataset

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def reformat(dataset, labels):
    dataset = dataset.reshape(-1, image_size, image_size, num_channels).astype(np.float32)
    labels = np.arange(num_classes)== labels[:,None].astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])

patch_size = 5
depth = 16
num_hidden = 64
beta = 0.001

graph = tf.Graph()

with graph.as_default():
    tf_train_dataset = tf.constant(train_dataset)
    tf_train_labels = tf.constant(train_labels)
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)


    layer_1_weight = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev = 0.1))
    layer_1_biases = tf.Variable(tf.zeros([depth]))

    layer_2_weight = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev = 0.1))
    layer_2_biases = tf.Variable(tf.zeros([depth]))

    layer_3_weight = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev = 0.1))
    layer_3_biases = tf.Variable(tf.zeros([num_hidden]))

    layer_4_weight = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev = 0.1))
    layer_4_biases = tf.Variable(tf.zeros([num_classes]))

    def model(data):
        conv = tf.nn.conv2d(data, layer_1_weight, [1,2,2,1], padding="SAME")
        hidden = tf.nn.relu(conv+layer_1_biases)

        conv = tf.nn.conv2d(hidden, layer_2_weight, [1,2,2,1], padding="SAME")
        hidden = tf.nn.relu(conv+layer_2_biases)

        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1]*shape[2]*shape[3]]) #Flatten 
        hidden = tf.nn.relu(tf.matmul(reshape,layer_3_weight)+ layer_3_biases)
        return tf.matmul(hidden,layer_4_weight) + layer_4_biases

    logits= model(tf_train_dataset)
    regularizers = tf.nn.l2_loss(layer_1_weight) + \
                   tf.nn.l2_loss(layer_2_weight) + \
                   tf.nn.l2_loss(layer_3_weight) + \
                   tf.nn.l2_loss(layer_4_weight)
    

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels = tf_train_labels))
    regularized_loss = tf.reduce_mean(loss + (beta * regularizers))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.03).minimize(regularized_loss)
    train_predictions = tf.nn.softmax(model(tf_train_dataset))
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

iters =100
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized Graph")
    
    for iteration in range(iters) :
        _,l,prediction = session.run([optimizer, loss, train_predictions])
        
        print("Loss at step %d is %f" %(iteration, l))
        print("Train Accuracy : %f %%" % accuracy(prediction, train_labels))
        print("Valid Accuracy : %f %%" % accuracy(valid_prediction.eval(), valid_labels))

    print("Final Accuracy : %f %%" %accuracy(test_prediction.eval(), test_labels) )