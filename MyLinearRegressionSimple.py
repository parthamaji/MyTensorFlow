import numpy as np
import matplotlib.pyplot as plt


num_points = 1000
dataset_point = []

for i in xrange(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    dataset_point.append([x1, y1])

x_data = [v[0] for v in dataset_point]
y_data = [v[1] for v in dataset_point]

plt.plot(x_data, y_data, "ro", label='Original data')
plt.legend()
plt.show()



import tensorflow as tf

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for step in xrange(16):
    sess.run(train)
    plt.plot(x_data, y_data, 'ro')
    plt.plot(x_data, sess.run(y))
    plt.xlabel('x')
    plt.xlim(-2,2)
    plt.ylim(0.1,0.6)
    plt.ylabel('y')
    plt.legend()
    plt.show()
    print(step, sess.run(W), sess.run(b))

