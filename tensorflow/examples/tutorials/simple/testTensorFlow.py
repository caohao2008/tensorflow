import tensorflow as tf
import numpy as np

#generate data array, 100 * 2
x_data = np.float32(np.random.rand(2, 100)) # 随机输入
print(x_data)
#generate data array , 2 * 1
y_data = np.dot([0.100, 0.200], x_data) + 0.300

print(y_data)

#b is bias,  
b = tf.Variable(tf.zeros([1]))
print(tf)
# w is weight
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
# y is result
# matmul: matrix multiply
y = tf.exp(tf.matmul(W, x_data) + b)

# loss is square loss
#all possible loss pls see doc: http://www.tensorfly.cn/tfdoc/api_docs/python/math_ops.html
#loss = tf.reduce_mean(tf.square(y - y_data))
loss = tf.reduce_mean(tf.abs(y - y_data))
# use gradient descent to train model
#all optimizer pls see doc: http://www.tensorfly.cn/tfdoc/api_docs/python/train.html
loss = tf.reduce_mean(tf.square(y - y_data))
#optimizer = tf.train.GradientDescentOptimizer(0.5)
optimizer = tf.train.AdagradOptimizer(0.5)
#optimizer = tf.train.FtrlOptimizer(0.5)
train = optimizer.minimize(loss)

# init tensor flow all variables
# variable related pls see doc : http://www.tensorfly.cn/tfdoc/api_docs/python/state_ops.html
init = tf.initialize_all_variables()

# 启动图 (graph)
# about session pls see doc : http://www.tensorfly.cn/tfdoc/api_docs/python/client.html#Session
sess = tf.Session()
sess.run(init)

# 拟合平面
for step in range(0, 2010):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))
