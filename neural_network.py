from decimal import Decimal
from itertools import pairwise
from numpy import exp
from random import randint


debug = True


def convert_to_string(continent_id):
    return {
        0: 'Europe',
        1: 'Asia',
        2: 'North America',
        3: 'South America',
        4: 'Africa',
        5: 'Asia',
        6: 'Australasia',
    }[continent_id]


def convert_continent_id_to_vector(continent_id):
    return {
        0: [1, 0, 0, 0, 0, 0, 0],
        1: [0, 1, 0, 0, 0, 0, 0],
        2: [0, 0, 1, 0, 0, 0, 0],
        3: [0, 0, 0, 1, 0, 0, 0],
        4: [0, 0, 0, 0, 1, 0, 0],
        5: [0, 0, 0, 0, 0, 1, 0],
        6: [0, 0, 0, 0, 0, 0, 1],
    }[continent_id]



class Neuron:
    def __init__(self, neuron_id):
        self.id = neuron_id
        self.weights = []
        for i in range(50):
            self.weights.append(randint(-5, 5))
        self.bias = randint(-5, 5)
        self.inputs = []
        self.output = None
        self.expected_output = None

    def set_input(self, input_vector):
        self.inputs = input_vector

    def get_output(self):
        activation = sum([Decimal(i) * Decimal(w) for i, w in zip(self.inputs, self.weights)]) + self.bias
        
        # Get the activation into the range -1 to 1.
        output = (Decimal(2) / (1 + exp(-activation))) - 1

        if debug:
            print(f'    Neuron {self.id}: {output}')
        self.output = output
        return output

    def update_weights_and_biases(self):
        cost = (self.output - self.expected_output) ** 2
        if debug:
            print(f'Cost: {cost}')


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
        if debug:
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

    def run(self, input_data):
        if debug:
            print(f'Input = {input_data}')
        self.layers[0].set_inputs(input_data)

        for layer in self.layers:
            layer.run()

    def train(self, data):
        for item in data:
            inputs = item[:-1]

            self.layers[0].set_inputs(inputs)

            for layer in self.layers:
                layer.run()

            # Backpropagation.
            expected_output = convert_continent_id_to_vector(item[-1])
            outputs = self.get_outputs()

            for layer in reversed(self.layers):
                for i, neuron in enumerate(layer.neurons):
                    neuron.expected_output = expected_output[i]
                    neuron.update_weights_and_biases()

    def test(self, data):
        success_count = 0
        total_count = 0
        for item in data:
            inputs = item[:-1]
            expected_output = item[-1]

            self.layers[0].set_inputs(inputs)

            for layer in self.layers:
                layer.run()

            outputs = self.get_outputs()
            output = outputs.index(max(outputs))

            if debug:
                print(
                    'Expected {}, got {}'.format(
                        convert_to_string(expected_output),
                        convert_to_string(output)
                    )
                )

            total_count += 1
            if output == expected_output:
                success_count += 1
        if debug:
            print('Success rate: {}%'.format(success_count * 100 / total_count))


    def get_outputs(self):
        last_layer = self.layers[-1]
        outputs = []
        for neuron in last_layer.neurons:
            outputs.append(neuron.output)
        return outputs

    def print_guess(self, outputs):
        continents = ['Europe', 'Asia', 'North America', 'South America', 'Africa', 'Asia', 'Australasia']
        for continent, output in zip(continents, outputs):
            output += 1
            output /= 2
            output = output.quantize(Decimal('0.1'))
            output_display = '█' * int((output / Decimal('0.1')))
            print(f'{continent: >13} {output_display}')

training_data = []
with open('training_data.txt', 'r') as f:
    for line in f:
        line = line.split(',')
        training_data.append(
            [Decimal(line[0]), Decimal(line[1]), int(line[2])]
        )

testing_data = []
with open('testing_data.txt', 'r') as f:
    for line in f:
        line = line.split(',')
        testing_data.append(
            [Decimal(line[0]), Decimal(line[1]), int(line[2])]
        )

neural_network = NeuralNetwork([3, 3, 7])
neural_network.train(training_data)
neural_network.test(testing_data)