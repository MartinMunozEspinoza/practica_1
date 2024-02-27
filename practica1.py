import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.1):
        self.weights = np.random.rand(num_inputs + 1)  # +1 for the bias
        self.learning_rate = learning_rate

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]  # Bias
        return 1 if summation > 0 else 0

    def train(self, training_inputs, labels, max_epochs):
        for epoch in range(max_epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

    def test(self, test_inputs, test_labels):
        correct = 0
        total = len(test_labels)
        for inputs, label in zip(test_inputs, test_labels):
            prediction = self.predict(inputs)
            if prediction == label:
                correct += 1
        accuracy = correct / total
        print("Accuracy:", accuracy)

    def plot_decision_boundary(self, inputs, labels):
        x_min, x_max = np.min(inputs[:, 0]) - 1, np.max(inputs[:, 0]) + 1
        y_min, y_max = np.min(inputs[:, 1]) - 1, np.max(inputs[:, 1]) + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        Z = np.array([self.predict([x, y]) for x, y in np.c_[xx.ravel(), yy.ravel()]])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(inputs[:, 0], inputs[:, 1], c=labels, marker='o', edgecolors='black')
        plt.xlabel('Input 1')
        plt.ylabel('Input 2')
        plt.title('Decision Boundary')
        plt.show()

def read_data(file_path):
    df = pd.read_csv(file_path, header=None)
    inputs = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    return inputs, labels

def main():
    # Lectura de datos de entrenamiento y prueba
    training_file = "XOR_trn.csv"
    test_file = "XOR_tst.csv"
    training_inputs, training_labels = read_data(training_file)
    test_inputs, test_labels = read_data(test_file)

    # Parámetros de entrenamiento
    num_inputs = len(training_inputs[0])
    learning_rate = float(input("Ingrese la tasa de aprendizaje: "))
    max_epochs = int(input("Ingrese el número máximo de épocas de entrenamiento: "))

    # Entrenamiento del perceptrón
    perceptron = Perceptron(num_inputs, learning_rate)
    perceptron.train(training_inputs, training_labels, max_epochs)

    # Prueba del perceptrón en datos de prueba
    perceptron.test(test_inputs, test_labels)

    # Mostrar gráficamente los patrones y la recta que los separa
    perceptron.plot_decision_boundary(training_inputs, training_labels)

if __name__ == "__main__":
    main()
