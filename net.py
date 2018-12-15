import numpy as np

class node():
    def __init__(self, num_weights):
        np.random.seed(1)
        #the shape of the weight matrix is linear, with length of the axis = num_weights
        self.weights = np.random.random((num_weights, 1))
    
    def activation(self, x):
        #We shall use sigmoid function for now; other functions will be implemented later (issue 1)
        return 1/(1+np.exp(-x))

    def run(self, inputs):
        #run the machine on current weight matrix
        return self.activation(np.dot(inputs, self.weights))

    #to train this node, we need derviative of the act. func. in the process, so we define (issue 2)

    def derivative_activation(self, x):
        #for now, we shall use the derivate of the sigmoid only
        return x * (1 - x)

    def train(self, training_set, num_epochs):
        for epoch in range(num_epochs):
            prediction = self.run(training_set['input_set'])

            #a simple (y-y1) error function. may also use mean-squared
            error = training_set['output_set'] - prediction

            #f(x+dx) = f(x) + dx*f'(x) where f'(x) = d(f(x))
            weight_increment = np.dot(training_set['input_set'].T, error * self.activation(prediction))
            #update of the weights
            self.weights += weight_increment

        




if __name__ == "__main__":
    """ usage tips:
        1. create a neural net object (it's already randomly initialized)
        2. prepare the training set (test set too, if necessary)
        3. train the network on the training set
        4. test the trained network on test tuples
    
    """
    
    num_features = 4
    neuron = node(num_features)

    #training_set dictionary
    #our input-outputs are even-odd numbers
    training_set = {}
    input_set = np.array([[0,0,0,0], [0,0,0,1], [0,0,1,0], [0,0,1,1], [0,1,0,0], [0,1,1,0]])
    output_set = np.array([[0,1,0,1,0,0]]).T
    training_set['input_set'] = input_set
    training_set['output_set'] = output_set

    #train it
    num_epochs = 100000
    neuron.train(training_set, num_epochs)

    #test tuple is the binary number 0111
    test_tuple = np.array([[0,1,0,1]])
    print(neuron.run(test_tuple))

    



