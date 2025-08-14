import numpy as np

class NeuralNetwork:
    def __init__(self, layers, input_size=10, output_size=1):
        self.layers = layers
        self.weights = [] # includes bias
        self.input_size = input_size
        self.output_size = output_size
        self.initialize()

    def initialize(self):
        self.weights.append(np.zeros((self.input_size, 1)))
        for i in range(1, len(self.layers)):
            # Initialize weights with small random values
            thisWeights = np.random.rand(self.layers[i - 1], self.layers[i]) * 0.01
            self.weights.append(thisWeights)

    # x is matrix, y is a vector
    def train(self, x, y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            self.train_epoch(x, y, learning_rate)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {self.mean_loss(x, y)}")
        print("Training complete.")

    # x is matrix, y is a vector
    def train_epoch(self, x, y, learning_rate=0.01):
        for i in range(x.shape[0]):
            all_x = self.forward(x[i])
            # back prop
            self.back(all_x, y[i], learning_rate)

    def forward(self, x):
        all_x = [x]
        for i in range(1, len(self.layers)):
            new_x = self.activation(np.dot(self.weights[i].T, x))
            x = new_x
            all_x.append(new_x)
        return all_x
 
    def back(self, all_x, y, learning_rate=0.01):
        delta = np.array([(all_x[-1] - y) * (1 - all_x[-1] ** 2)]).reshape(-1, 1)
        # print (f"all x shape: {[np.shape(x) for x in all_x]}")
        for i in range (len(self.layers) - 1, 0, -1):
            # print (f"Delta shape for layer {i}: {np.shape(delta)}")
            # print (f"all_x shape for layer {i}: {np.shape(all_x[i - 1].reshape(-1, 1))}")
            # Calculate weight gradient
            weight_gradmat = delta @ all_x[i - 1].reshape(-1, 1).T
            # print (f"Weight gradient for layer {i}: {np.shape(weight_gradmat)}")
            # print (f"weights before update for layer {i}: {np.shape(self.weights[i])}")
            self.weights[i] -= learning_rate * weight_gradmat.T
            delta = (self.weights[i] @ delta) * (1 - all_x[i - 1].reshape(-1, 1) ** 2)

    def activation(self, s):
        return self.tanh(s)
    
    def ddx_activation(s):
        return self.ddx_tanh(s)

    def tanh(self, s):
        return np.tanh(s)
    
    def ddx_tanh(self, s):
        return 1 - np.tanh(s) ** 2

    # x is a matrix, y is a vector
    def mean_loss(self, x, y):
        outputs = np.array([self.forward(x[i])[-1] for i in range(x.shape[0])]).flatten()
        return np.mean((outputs - y) ** 2)

if __name__ == "__main__":
    nn = NeuralNetwork([10, 20, 1])
    x = np.random.rand(100, 9)
    x = np.hstack([x, np.ones((100, 1))])  # Add bias column of 1s
    y = np.random.rand(100, 1)
    nn.train(x, y, epochs=8000, learning_rate=0.00001)
    print("Final loss:", nn.mean_loss(x, y))
    print("Weights:", nn.weights)