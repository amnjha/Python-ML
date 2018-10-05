import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf

import os

data_root = "/Users/amajha/Documents/ml_data"
pickle_file = 'notMNIST.pickle'
image_size = 28
num_classes = 10

train_dataset = None
test_dataset = None
train_labels = None
test_labels = None
valid_dataset = None
valid_labels = None

#load pickled data
def load_pickle(filename):
    pickle_file = os.path.join(data_root,filename)
    global train_dataset
    global test_dataset
    global train_labels
    global test_labels
    global valid_dataset
    global valid_labels
    with open(pickle_file, 'rb') as pk:
        dataset =  pickle.load(pk)
        train_dataset = dataset['train_dataset']
        train_labels = dataset['train_labels']
        test_dataset = dataset['test_dataset']
        test_labels = dataset['test_labels']
        valid_dataset = dataset['valid_dataset']
        valid_labels = dataset['valid_labels']

load_pickle(pickle_file)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def reformat_data(dataset, labels):
    dataset.shape = (len(dataset), image_size*image_size)
    labels = (np.arange(num_classes) == labels[:,None]).astype(float)
    return dataset, labels

def randomize(dataset, labels):
    permutation = np.random.permutation(len(dataset))
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_lables = labels[permutation]
    return shuffled_dataset, shuffled_lables

print("\nReformatting Data")
train_dataset, train_labels = reformat_data(train_dataset, train_labels)
test_dataset, test_labels = reformat_data(test_dataset, test_labels)
valid_dataset, valid_labels = reformat_data(valid_dataset, valid_labels)
print("Reformat Complete\n")


train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

train_subset = 20000

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.constant(train_dataset[:train_subset,:])
    tf_train_labels = tf.constant(train_labels[:train_subset])
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    weights = tf.Variable(tf.truncated_normal([image_size*image_size, num_classes]))
    biases =  tf.Variable(tf.zeros([num_classes]))

    logits = tf.matmul(tf_train_dataset, weights) +  biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = tf_train_labels, logits = logits))

    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    train_predictions =  tf.nn.softmax(logits)
    valid_predictions = tf.nn.softmax(tf.matmul(tf_valid_dataset,weights)+biases)
    test_predictions = tf.nn.softmax(tf.matmul(tf_test_dataset,weights)+biases)

num_steps = 801

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        _, l, prediction  = sess.run([optimizer, loss, train_predictions])
        if (step%100==0):
            print("Training Accuracy: %.lf%%" %accuracy(train_predictions.eval(), train_labels))
            print("Loss at step %d : %f " % (step, l))
            print("Validation Accuracy: %.lf%%" %accuracy(valid_predictions.eval(), valid_labels))
    print("Test Accuracy: %.lf%%" %accuracy(test_predictions.eval(), test_labels))