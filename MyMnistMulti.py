from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
import sys
import os.path
import argparse

#try:
#    log_subdir = sys.argv[1]
#except IndexError:
#    print "Usage: python filename.py run_dir"
#    sys.exit()

#Parser
parser = argparse.ArgumentParser()
parser.add_argument("--mode", required = True, help = "options: init (trains from scratch), load (loads a checkpoint)")
parser.add_argument("--logdir", required = True, help = "logdir for summary and checkpoint")
args = parser.parse_args()
#if not any(args):
#    parse.error('No arguments provided! Try --help')
#print args.init
#print args.logdir

#Define Flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', "data")
mnist_train = os.path.join("/local/scratch/ppm27/mnist_multi", args.logdir)
tf.app.flags.DEFINE_string('train_dir', mnist_train, "Directory where to write event logs and checkpoint.")
checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')

###############################################################################
# Set up + Dataset
###############################################################################
#Import data
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

#Define model
with tf.name_scope('DataSetXY'):
    x = tf.placeholder(tf.float32, [None, 784], "X")
    y_actual = tf.placeholder(tf.float32, [None, 10], "Y")
    x_image = tf.reshape(x, [-1,28,28,1])
    #print "Shape(x_image): " , x_image

#Define weight and bias variable
def weig_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.05)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

#Define layers templates
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

###############################################################################
# Define Model
###############################################################################
##Layer 1
with tf.name_scope('Conv1'):
    W_conv1 = weig_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

##Layer 2
with tf.name_scope('Conv2'):
    W_conv2 = weig_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

##Layer 3
with tf.name_scope('FC1'):
    W_fc1 = weig_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

##Dropout
with tf.name_scope('DropOut'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

##Layer 4
with tf.name_scope('FC2'):
    W_fc2 = weig_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_predict = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

###############################################################################
# Define Cost + Accuracy
###############################################################################
#Loss
with tf.name_scope('Xentropy'):
    cross_entropy = -tf.reduce_sum(y_actual * tf.log(y_predict))

with tf.name_scope('TrainStep'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_actual, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

###############################################################################
# Set up Session + TensorBoard
###############################################################################
#Create a Session

sess = tf.Session()
saver = tf.train.Saver()
if args.mode == "init":
    sess.run(tf.initialize_all_variables())
elif args.mode == "load":    
    saver.restore(sess, checkpoint_path)
else:
    print("Session mode argument missing!")

#TensorBoard
tf.histogram_summary('W_conv1', W_conv1)
tf.histogram_summary('b_conv1', b_conv1)
tf.histogram_summary('h_conv1', h_conv1)
tf.histogram_summary('W_conv2', W_conv1)
tf.histogram_summary('b_conv2', b_conv1)
tf.histogram_summary('h_conv2', h_conv1)
tf.histogram_summary('W_fc1', W_fc1)
tf.histogram_summary('b_fc1', b_fc1)
tf.histogram_summary('h_fc1', h_fc1)
tf.histogram_summary('W_fc2', W_fc1)
tf.histogram_summary('b_fc2', b_fc1)
tf.histogram_summary('h_fc2', h_fc1)
tf.scalar_summary('accuracy', accuracy)
tf.scalar_summary('cross_entropy', cross_entropy)

#Image Summary Experiment
with tf.name_scope('VIS_Conv1'):
    vis_h_conv1 = tf.slice(h_conv1, (0, 0, 0, 0), (1, -1, -1, -1), name='slice_first_input')
    vis_h_conv1 = tf.reshape(vis_h_conv1, (28, 28, 32))
    vis_h_conv1 = tf.transpose(vis_h_conv1, (2, 0, 1))
    vis_h_conv1 = tf.reshape(vis_h_conv1, (-1, 28, 28, 1))
    tf.image_summary("first_conv", vis_h_conv1, max_images=1)

with tf.name_scope('VIS_Conv2'):
    vis_h_conv2 = tf.slice(h_conv2, (0, 0, 0, 0), (1, -1, -1, -1), name='slice_first_input')
    vis_h_conv2 = tf.reshape(vis_h_conv2, (14, 14, 64))
    vis_h_conv2 = tf.transpose(vis_h_conv2, (2, 0, 1))
    vis_h_conv2 = tf.reshape(vis_h_conv2, (-1, 14, 14, 1))
    tf.image_summary("second_conv", vis_h_conv2, max_images=1)

merged_summary = tf.merge_all_summaries()
writer_summary = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph.as_graph_def(add_shapes=True))

###############################################################################
# Train + Test
###############################################################################
#Train and report training accuracy
for i in range(100):
    batch_xs, batch_ys = mnist.train.next_batch(32)
    sess.run(train_step, feed_dict={x: batch_xs, y_actual :batch_ys, keep_prob: 0.5})
    if i % 10 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_actual :batch_ys, keep_prob: 1.0})
        print("Step %d, training accuracy %g"%(i, train_accuracy))
        summary = sess.run(merged_summary, {x: batch_xs, y_actual : batch_ys, keep_prob: 1.0})
        writer_summary.add_summary(summary, i)
        #print(np.linalg.norm(sess.run(W_fc2)))

#Save parameters at the end of training
save_path = saver.save(sess, checkpoint_path)
print("Model saved in file: %s" % save_path)
 
#Measure accuracy on test images
print("running test accuracy..")
print("test accuracy %g"% \
       sess.run(accuracy, feed_dict={x: mnist.test.images, y_actual: mnist.test.labels, keep_prob: 1.0}))
