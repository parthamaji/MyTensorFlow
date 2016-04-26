from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

#Define Flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', "data")
flags.DEFINE_string('summaries_dir', '/tmp/mnist_logs/', "logs")

#Import data
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

#Define model
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

with tf.name_scope('Wx_b'):
  y = tf.nn.softmax(tf.matmul(x, W) + b)


#Define cost function
with tf.name_scope('xent'):
  cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
with tf.name_scope('train'):
  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#Prediction test
with tf.name_scope('test'):
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Create a Session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#TensorBoard
#tf.histogram_summary('weights', W)
#tf.histogram_summary('bias', b)
#tf.histogram_summary('y', y)
tf.scalar_summary('accuracy', accuracy)
tf.scalar_summary('xent', cross_entropy)

merged_summary = tf.merge_all_summaries()
writer_summary = tf.train.SummaryWriter(FLAGS.summaries_dir, sess.graph.as_graph_def(add_shapes=True))

#Train model
for i in range(100000):
  batch_xs, batch_ys = mnist.train.next_batch(225)
  sess.run(train_step, {x: batch_xs, y_ :batch_ys})
  summary = sess.run(merged_summary, {x: batch_xs, y_ : batch_ys})
  writer_summary.add_summary(summary, i)

#Measure accuracy
accuracy = sess.run(accuracy, {x: mnist.test.images, y_: mnist.test.labels})
print(accuracy)

