import numpy as np
import os
import sys

# cur_dir = "/home/ray/code/NNs/Simple-AI-Framework/"
# sys.path.append(cur_dir)

import similarflow as sf
import matplotlib.pyplot as plt

red_points = np.random.randn(50, 2) - 2 * np.ones((50, 2))
blue_points = np.random.randn(50, 2) + 2 * np.ones((50, 2))

X = sf.Placeholder()
y = sf.Placeholder()
W = sf.Variable(np.random.randn(2, 2))
b = sf.Variable(np.random.randn(2))

p = sf.softmax(sf.add(sf.matmul(X, W), b))
loss = sf.negative(sf.reduce_sum(sf.reduce_sum(sf.multiply(y, sf.log(p)), axis=1)))

train_op = sf.train.GradientDescentOptimizer(0.01).minimize(loss)

feed_dict = {
    X: np.concatenate((red_points, blue_points)),
    y: [[1, 0]] * len(blue_points) + [[0, 1]] * len(red_points),
}

with sf.Session() as sess:
    for step in range(1000):
        loss_value = sess.run(loss, feed_dict=feed_dict)
        if step % 10 == 0:
            print("Step:", step, "Loss:", loss_value)
        sess.run(train_op, feed_dict=feed_dict)
    W_value = sess.run(W)
    b_value = sess.run(b)
    print("W:", W_value)
    print("b:", b_value)

x_axis = np.linspace(-4, 4, 100)
y_axis = -W_value[0][0] / W_value[1][0] * x_axis - b_value[0] / W_value[1][0]
plt.plot(x_axis, y_axis)
plt.scatter(red_points[:, 0], red_points[:, 1], color="red")
plt.scatter(blue_points[:, 0], blue_points[:, 1], color="blue")
plt.savefig("./exam_simiflow/perceptron.pdf")
