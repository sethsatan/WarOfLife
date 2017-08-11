import tensorflow as tf
from matplotlib import pyplot as plt

shape = (30, 30)
initial_board = tf.random_uniform(shape, minval=0, maxval=2, dtype=tf.int32)
print(initial_board)
with tf.Session() as session:
    X = session.run(initial_board)
    print(X)

fig = plt.figure()
plot = plt.imshow(X, cmap='Greys',  interpolation='nearest')
plt.show()


