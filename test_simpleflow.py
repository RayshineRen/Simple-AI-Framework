import numpy as np
import simpleflow as sf
import matplotlib.pyplot as plt

input_x = np.linspace(-1, 1, 100)
input_y = input_x * 3 + np.random.randn(input_x.shape[0]) * 0.5

# placeholder for training data
x = sf.placeholder()
y_ = sf.placeholder()

w = sf.Variable([[1.0]], name="weight")
b = sf.Variable(0.0, name="threshold")

y = w * x + b

loss = sf.reduce_sum(sf.square(y - y_))

train_op = sf.GradientDescentOptimizer(learning_rate=0.005).minimize(loss)

feed_dict = {x: np.reshape(input_x, (-1, 1)), y_: np.reshape(input_y, (-1, 1))}
feed_dict = {x: input_x, y_: input_y}

with sf.Session() as sess:
    for step in range(100):
        loss_value = sess.run(loss, feed_dict=feed_dict)
        mse = loss_value / input_x.shape[0]
        if step % 5 == 0:
            print("step: {}, mse: {}".format(step, mse))
        sess.run(train_op, feed_dict=feed_dict)
    w_value, b_value = sess.run(w, feed_dict=feed_dict), sess.run(
        b, feed_dict=feed_dict
    )
    print("w: {}, b: {}".format(w_value, b_value))

w_value = float(w_value)
max_x, min_x = np.max(input_x), np.min(input_x)
max_y, min_y = w_value * max_x + b_value, w_value * min_x + b_value

plt.scatter(input_x, input_y)
plt.plot([max_x, min_x], [max_y, min_y], "r")
plt.savefig("test_simpleflow.pdf")
