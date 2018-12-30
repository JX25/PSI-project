import tensorflow as tf

# network parameters
n_input = 13
n_output = 1
n_hidden_1 = 10
n_hidden_2 = 10
# learning parameters
learning_const = 0.0001
number_epochs = 1000
batch_size = 100

X = tf.placeholder("float", n_input)
Y = tf.placeholder("float", n_output)
