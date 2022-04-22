from decimal import Decimal
from itertools import pairwise


class Neuron:
    def __init__(self, neuron_id):
        self.id = neuron_id
        self.inputs = []
        self.outputs = []

    def clear_inputs(self):
        self.inputs = []

    def set_input(self, input_vector):
        self.inputs = input_vector

    def get_output(self):
        output = sum(self.inputs)
        print(f'    Neuron {self.id}: {output}')
        return output


class Layer:
    def __init__(self, layer_id):
        self.id = layer_id
        self.neurons = []
        self.next_layer = None

    def add_neuron(self, n):
        self.neurons.append(n)

    def set_next_layer(self, l):
        self.next_layer = l

    def set_inputs(self, value):
        for n in self.neurons:
            n.set_input(value)

    def calculate_outputs(self):
        self.outputs = [n.get_output() for n in self.neurons]

    def set_next_layer_inputs(self, value=None):
        if self.next_layer is None:
            return

        if value is None:
            value = self.outputs
        for n in self.next_layer.neurons:
            n.set_input(value)

    def run(self):
        print(f'Layer {self.id}')
        self.calculate_outputs()
        self.set_next_layer_inputs()


class NeuralNetwork:
    def __init__(self, layer_spec):
        self.layers = []
        for i, neuron_count in enumerate(layer_spec):
            layer = Layer(i + 1)
            for j in range(neuron_count):
                neuron = Neuron(j + 1)
                layer.add_neuron(neuron)
            self.layers.append(layer)

        for layer, next_layer in pairwise(self.layers):
            layer.set_next_layer(next_layer)

    def run(self, inputs):
        print(f'Input = {inputs}')
        self.layers[0].set_inputs(inputs)

        for layer in self.layers:
            layer.run()

neural_network = NeuralNetwork([5, 2, 5, 1])
neural_network.run([1, 2, 3])