import numpy as np

class NeuralNetwork:
    def __init__(self, layers, input_size=10, output_size=1):
        self.layers = layers
        self.weights = [] # includes bias
        self.input_size = input_size
        self.output_size = output_size
        self.initialize()
        self.epsilon = 1e-5

    def initialize(self):
        self.weights.append(np.zeros((self.input_size, 1)))
        for i in range(1, len(self.layers)):
            # Initialize weights with small random values
            thisWeights = np.random.randn(self.layers[i], self.layers[i-1]) * np.sqrt(2/self.layers[i-1])
            self.weights.append(thisWeights)

    # x is matrix, y is a vector
    def train(self, x, y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            self.train_epoch(x, y, learning_rate)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {self.cross_entropy(x, y)}")
        print("Training complete.")

    # x is matrix, y is a vector
    def train_epoch(self, x, y, learning_rate=0.01):
        for i in range(x.shape[0]):
            all_x, preactivations = self.forward(x[i])
            # back prop
            self.back(all_x, y[i], preactivations, learning_rate)

    def forward(self, x):
        all_x = [x]
        preactivations = [x]
        for i in range(1, len(self.layers) - 1):
            signal = np.dot(self.weights[i], x)
            preactivations.append(signal)
            new_x = self.activation(signal)
            x = new_x
            all_x.append(new_x)
        # Last layer uses softmax
        preactivations.append(np.dot(self.weights[-1], x))
        new_x = self.softmax(np.dot(self.weights[-1], x))
        all_x.append(new_x)
        return all_x, preactivations
 
    def back(self, all_x, y, preactivations, learning_rate=0.01):
        delta = np.array([all_x[-1] - y]).reshape(-1, 1)
        # print (f"all x shape: {[np.shape(x) for x in all_x]}")
        for i in range (len(self.layers) - 1, 0, -1):
            # print (f"Delta shape for layer {i}: {np.shape(delta)}")
            # print (f"all_x shape for layer {i}: {np.shape(all_x[i - 1].reshape(-1, 1))}")
            # Calculate weight gradient
            weight_gradmat = delta @ all_x[i - 1].reshape(1, -1)
            # print (f"Weight gradient for layer {i}: {np.shape(weight_gradmat)}")
            # print (f"weights before update for layer {i}: {np.shape(self.weights[i])}")
            self.weights[i] -= learning_rate * weight_gradmat
            delta = (self.weights[i].T @ delta) * self.ddx_activation(preactivations[i - 1].reshape(-1, 1))

    def activation(self, s):
        return self.relu(s)

    def softmax(self, s):
        exp_s = np.exp(s - np.max(s)) # subtract max for bounding. gets cancelled anyway.
        return exp_s / np.sum(exp_s, axis=0, keepdims=True)

    def relu(self, s):
        return np.maximum(0, s)

    def ddx_activation(self, s):
        return self.ddx_relu(s)

    def ddx_relu(self, s):
        return np.where(s > 0, 1, 0)
    
    # x is a matrix, y is a matrix
    def cross_entropy(self, x, y):
        preds = np.array([self.forward(x[i])[0][-1].flatten() for i in range(x.shape[0])])
        return -np.sum(y * np.log(preds + self.epsilon)) / x.shape[0]

if __name__ == "__main__":
    nn = NeuralNetwork([10, 20, 20, 10, 10])
    x = np.random.rand(100, 9)
    x = np.hstack([x, np.ones((100, 1))])  # Add bias column of 1s
    y = np.eye(10)[np.random.choice(10, 100)]
    nn.train(x, y, epochs=8000, learning_rate=0.001)
    print("Final loss:", nn.cross_entropy(x, y))
    print("Weights:", nn.weights)