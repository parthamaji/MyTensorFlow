import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pandas as pd

num_clusters = 3
num_steps = 100
num_points = 2000
dataset_points = []

for i in xrange(num_points):
    if np.random.random() > 0.5:
        dataset_points.append([np.random.normal(0.0, 0.9), np.random.normal(0.0, 0.9)])
    else:
        dataset_points.append([np.random.normal(3.0, 0.5), np.random.normal(1.0, 0.5)])


x_data = [v[0] for v in dataset_points]
y_data = [v[1] for v in dataset_points]


plt.plot(x_data, y_data, 'r.', label='original data')
plt.legend()
plt.show()


#df = pd.DataFrame({"x": [v[0] for v in dataset_points],
#                   "y": [v[1] for v in dataset_points]})
#
#sns.lmplot("x", "y", data=df, fit_reg=False, size=7)
#plt.show()
 

vectors = tf.constant(dataset_points)
centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0,0], [num_clusters, -1]))
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroids = tf.expand_dims(centroids, 1)

print expanded_vectors.get_shape()
print expanded_centroids.get_shape()


distances = tf.reduce_sum(tf.square(tf.sub(expanded_vectors, expanded_centroids)), 2)
assignments = tf.argmin(distances, 0)

means = tf.concat(0, [tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])), reduction_indices = [1]) for c in xrange(num_clusters)])

update_centroids = tf.assign(centroids, means)
init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)

for step in xrange(num_steps):
   _, centroid_values, assignment_values = sess.run([update_centroids,
                                                    centroids,
                                                    assignments])
print "centroids"
print centroid_values

data = {"x": [], "y": [], "cluster": []}
for i in xrange(len(assignment_values)):
  data["x"].append(dataset_points[i][0])
  data["y"].append(dataset_points[i][1])
  data["cluster"].append(assignment_values[i])
df = pd.DataFrame(data)
sns.lmplot("x", "y", data=df, 
           fit_reg=False, size=7, 
           hue="cluster", legend=False)
plt.show()

