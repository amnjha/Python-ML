import numpy as np 
import tensorflow as tf
import pickle
import os

data_root = "/Users/amajha/Documents/ml_data"
pickle_file = 'notMNIST.pickle'
image_size = 28
num_classes = 10
batch_size = 128

train_dataset = None
test_dataset = None
train_labels = None
test_labels = None
valid_dataset = None
valid_labels = None

def load_pickle_files(filename):
    global train_dataset
    global test_dataset
    global valid_dataset
    global train_labels
    global test_labels
    global valid_labels

    with open(os.path.join(data_root, filename),'rb') as pk:
        dataset = pickle.load(pk)
        train_dataset = dataset['train_dataset']
        test_dataset = dataset['test_dataset']
        train_labels = dataset['train_labels']
        test_labels = dataset['test_labels']
        valid_dataset = dataset['valid_dataset']
        valid_labels = dataset['valid_labels']

def reformat(dataset,labels):
    dataset.shape = (len(dataset), image_size*image_size)
    labels = (np.arange(num_classes) == labels[:,None]).astype(float)

    return dataset, labels

def randomize(dataset,labels):
    permute = np.random.permutation(len(dataset))
    dataset = dataset[permute,:]
    labels = labels[permute]
    return dataset, labels

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

load_pickle_files(pickle_file)

#Reformat 
train_dataset, train_labels = reformat(train_dataset, train_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)

#Randomize
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

graph = tf.Graph()
with graph.as_default():

    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size*image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_classes))

    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    wieghts = tf.Variable(tf.truncated_normal([image_size*image_size, num_classes]))
    biases = tf.Variable(tf.zeros([num_classes]))

    logits = tf.matmul(tf_train_dataset, wieghts)+ biases

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = tf_train_labels, logits = logits))
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    train_predictions = tf.nn.softmax(tf.matmul(tf_train_dataset, wieghts)+biases)
    test_predictions = tf.nn.softmax(tf.matmul(tf_test_dataset, wieghts)+biases)
    valid_predictions = tf.nn.softmax(tf.matmul(tf_valid_dataset, wieghts)+biases)

num_steps = 3001
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialization Done!")

    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:offset+batch_size,:]
        batch_labels = train_labels[offset:offset+batch_size]

        feed_dict = {
            tf_train_dataset : batch_data, 
            tf_train_labels: batch_labels
        }

        _, l, predictions = session.run([optimizer, loss, train_predictions], feed_dict=feed_dict)

        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(valid_predictions.eval(), valid_labels))

    print("Test accuracy: %.1f%%" % accuracy(test_predictions.eval(), test_labels))