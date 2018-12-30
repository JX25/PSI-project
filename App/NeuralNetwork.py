import tensorflow as tf
import numpy as np

class ANN:
    def __init__(self):
        self.n_input = 13
        self.n_hidden_1 = 6
        self.n_hidden_2 = 5
        self.n_output = 1
        self.learning_constant = 0.001
        self.number_epochs = 1000
        self.batch_size = 13
        self.neural_network = None
        self.X = tf.placeholder("float", [None, self.n_input])
        self.Y = tf.placeholder("float", [self.n_output, None])

        # Biases first hidden layer
        self.b1 = tf.Variable(tf.random_normal([self.n_hidden_1]))
        # Biases second hidden layer
        self.b2 = tf.Variable(tf.random_normal([self.n_hidden_2]))
        # Biases output layer
        self.b3 = tf.Variable(tf.random_normal([self.n_output]))

        # Weights connecting input layer with first hidden layer
        self.w1 = tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1]))
        # Weights connecting first hidden layer with second hidden layer
        self.w2 = tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2]))
        # Weights connecting second hidden layer with output layer
        self.w3 = tf.Variable(tf.random_normal([self.n_hidden_2, self.n_output]))

    def multilayer_perceptron(self, input_d):
        # Task of neurons of first hidden layer
        layer_1 = tf.nn.relu(tf.add(tf.matmul(input_d, self.w1), self.b1))
        # Task of neurons of second hidden layer
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, self.w2), self.b2))
        # Task of neurons of output layer
        out_layer = tf.add(tf.matmul(layer_2, self.w3), self.b3)
        return out_layer

    def train_model(self, train_x, train_y, input_x, input_y):
        self.neural_network = self.multilayer_perceptron(self.X)
        # Define the loss or the error
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.neural_network, labels=self.Y))
        # Define how to fix it
        optimizer = tf.train.GradientDescentOptimizer(self.learning_constant).minimize(loss_op)
        # Initializing the variables
        init = tf.global_variables_initializer()
        # Create a session
        with tf.Session() as sess:
            sess.run(init)
            # Training epoch
            for epoch in range(self.number_epochs):
                # Get one batch of images
                # batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run the optimizer feeding the network with the batch
                sess.run(optimizer, feed_dict={self.X: train_x, self.Y: train_y})
                # Display the epoch (just every 100)
                if epoch % 100 == 0:
                    print("Epoch:", '%d' % epoch)
            for (x, y) in zip(input_x, np.matrix.transpose(input_y)):
                pred = tf.nn.tanh(self.neural_network)  # Apply softmax to logits
                correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.Y, 1))
                # Calculate accuracy
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                print("Accuracy:", accuracy.eval({self.X: x.reshape(1, 13), self.Y: y.reshape(1, 1)}))