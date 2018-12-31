import tensorflow as tf
import numpy as np
from App.main import facts, facts_vtp, votes_to_predict, votes
x_train = facts
y_train = np.reshape(votes, (1, 6))
x_predict = facts_vtp
y_predict = votes_to_predict
'''
[[0.44379274 0.33497492 0.57438354 0.33577876 0.78553719 0.59079002
  0.62636601 0.47310321 0.39331029 0.42916541 0.290212   0.39825435
  0.54639186 0.24324136 0.23614244 0.36548388 0.42756726 0.56506985
  0.50647341 0.38420563 0.48495846 0.41896208 0.81963867 0.67295304
  0.46116349 0.35971053 0.80387133 0.77217117 0.60407348 0.41310295
  0.36173837 0.28745767 0.66614291 0.58054518 0.64687567 0.32503791
  0.67087339 0.39847462 0.27678091 0.14031286 0.33981144 0.35145887
  0.7888856  0.83048082 0.36018492 0.07751938]]
  ************************************************************
  [0.41002182, 0.33106521, 0.59460954, 0.33997453, 0.77888681, 0.55970833,
   0.61547523, 0.47387763, 0.38288054, 0.4171104 , 0.26244765, 0.37401788,
    0.43214673, 0.26636168, 0.36105934, 0.37590674, 0.42378748, 0.55342133,
     0.52323365, 0.37748305, 0.47595316, 0.39160258, 0.82663576, 0.35975504,
      0.44569178, 0.33503275, 0.77803847, 0.76463756, 0.591683  , 0.36567251,
       0.44779146, 0.33833351, 0.6223768 , 0.5304529 , 0.65787249, 0.37748081,
        0.63729392, 0.36157092, 0.26976068, 0.19622761, 0.3445407 , 0.37363874,
        0.75921605, 0.82683563, 0.33908002, 0.17685821]
  
 *********************************  
  [[0.39331029 0.36173837 0.60407348 0.33497492]]
  *********************************
  [0.38288054, 0.44779146, 0.591683  , 0.33106521]
'''
# mlp parameters
n_input = 13
n_hidden_1 = 3
n_hidden_2 = 8
n_hidden_3 = 6
n_hidden_4 = 4
n_output = 1

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [n_output, None])

# weights
w1 = tf.Variable(tf.random_uniform([n_input, n_hidden_1], -1.0, 1.0))
w2 = tf.Variable(tf.random_uniform([n_hidden_1, n_hidden_2], -1.0, 1.0))
w3 = tf.Variable(tf.random_uniform([n_hidden_2, n_hidden_3], -1.0, 1.0))
w4 = tf.Variable(tf.random_uniform([n_hidden_3, n_hidden_4], -1.0, 1.0))
w5 = tf.Variable(tf.random_uniform([n_hidden_4, n_output], -1.0, 1.0))

# bias
b1 = tf.Variable(tf.zeros([n_hidden_1]), name="bias1")
b2 = tf.Variable(tf.zeros([n_hidden_2]), name="bias2")
b3 = tf.Variable(tf.zeros([n_hidden_3]), name="bias3")
b4 = tf.Variable(tf.zeros([n_hidden_4]), name="bias3")
b5 = tf.Variable(tf.zeros([n_output]), name="bias3")

# layers
layer1 = tf.tanh(tf.matmul(X, w1) + b1)
layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2)
layer3 = tf.tanh(tf.matmul(layer2, w3) + b3)
layer4 = tf.sigmoid(tf.matmul(layer3, w4) + b4)
out = tf.tanh(tf.matmul(layer4, w5) + b5)

cost = tf.reduce_mean(-Y*tf.log(out) - (1-Y)*tf.log(1-out))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0005).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(init)

    for i in range(100000):
        session.run(optimizer, feed_dict={X: x_train, Y: y_train})
        if i % 1000 == 0:
            print(str(session.run(cost, feed_dict={X: x_train, Y: y_train})))

    answear = tf.equal(tf.floor(out + 0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(answear, "float"))
    print(str(answear) + " " + str(accuracy))
    print(str(session.run([out], feed_dict={X: x_train, Y: y_train})))
    print("Accuracy " + str(accuracy.eval({X: x_train, Y: y_train})*100))
