import tensorflow as tf


class ANN:
    # network parameters
    def __init__(self):
        self.n_input = 13
        self.n_output = 1
        self.n_hidden_1 = 10
        self.n_hidden_2 = 10
        # learning parameters
        self.learning_const = 0.0001
        self.number_epochs = 1000
        self.batch_size = 100

        self.X = tf.placeholder("float", self.n_input)
        self.Y = tf.placeholder("float", self.n_output)

        #biases for hidden1, hidden2, output
        self.b1 = tf.Variable(tf.random_normal([self.n_hidden_1]))
        self.b2 = tf.Variable(tf.random_normal([self.n_hidden_2]))
        self.b3 = tf.Variable(tf.random_normal([self.n_output]))

        #weights connecting between layers
        self.w1 = tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1]))
        self.w2 = tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2]))
        self.w3 = tf.Variable(tf.random_normal([self.n_hidden_2, self.n_output]))

    def multilayer_perceptron(self, input_data): #model
        self.layer_1 = tf.nn.tanh(tf.add(tf.matmul(input_data, self.w1), self.b1))
        self.layer_2 = tf.nn.tanh(tf.add(tf.matmul(self.layer_1, self.w2), self.b2))
        self.out_layer = tf.add(tf.matmul(self.layer_2, self.w3), self.b3)
        return self.out_layer


def training(neural_network : ANN, Y, learn_const, batch_x, batch_y):
    #loss of the error
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neural_network, labels=Y))
    #fix it
    optimizer = tf.train.GradientDescentOptimizer(learn_const).minimize(loss_op)

    init = tf.global_variables_initializer()
    #create session
    with tf.Session() as sess:
        sess.run(init)
        #training epoch
        for epoch in range(neural_network.number_epochs):
            sess.run(optimizer, feed_dict={neural_network.X: batch_x, neural_network.Y: batch_y})
            if epoch % 100 == 0:
                print("Epoch:", '%d' % (epoch))

    return neural_network

def test_model(neural_network : ANN, input, output):
    pred = tf.nn.softmax(neural_network)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(output, 1))
    # accuracy
    accuracy = tf.reduce_mean( tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({neural_network.X: input, neural_network.Y: output}))
