import sys
import json
import math
from copy import deepcopy
from random import random, shuffle

import numpy as np

class Layer(object):
    
    def __init__(self, n_inputs=0, n_neurons=0, activation='relu'):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.activation_type = activation
        
        if n_inputs == 0 or n_neurons == 0:
            return None
        
        self.random()
        self.update()
    
    def random(self):
        # Create matrix of shape (n_inputs, n_neurons) with normal dist. multiplied by 10%
        self.weights = np.random.randn(self.n_inputs, self.n_neurons) * 0.1
        # Create matrix of zeros
        self.biases = np.zeros((1, self.n_neurons))
    
    def update(self):
        try:
            activation_type = ACTIVATION_LOOKUP[self.activation_type]
        except KeyError:
            valid_types = list(ACTIVATION_LOOKUP.keys())
            raise Exception("Unknown activation '%s': expected %r" % (activation_type, valid_types))
        self.activation_fn = activation_type['forward']
        self.backward_fn = activation_type['backward']
        self.n_inputs = len(self.weights)
        self.n_neurons = len(self.biases[0])
    
    def fire(self, inputs):
        '''
        inputs list of inputs; can accept batches
        '''
        self.inputs = np.array(inputs)
        # Multiply inputs by weights, then add biases
        self.outputs_noa = np.dot(inputs, self.weights) + self.biases
        
        self.outputs = self.activation(self.outputs_noa)
        # outputs batches
        return self.outputs
    
    def activation(self, inputs):
        return self.activation_fn(inputs)
    
    def copy(self):
        new_layer = Layer(n_inputs=0, n_neurons=0)
        new_layer.activation_type = self.activation_type
        new_layer.weights = self.weights.copy()
        new_layer.biases = self.biases.copy()
        new_layer.update()
        return new_layer
    
    def fromjson(self, json_data):
        self.activation_type = json_data.get('activation', self.activation_type)
        self.weights = np.array(json_data['weights'])
        self.biases = np.array(json_data['biases'])
        self.update()
    
    def tojson(self):
        json_data = {}
        json_data['activation'] = self.activation_type
        json_data['weights'] = self.weights.tolist()
        json_data['biases'] = self.biases.tolist()
        return json_data
    
    def a_relu(x):
        '''
        ReLU (Rectified Linear Unit)
        '''
        y = np.maximum(0, x)
        return y
        
    def a_relu_derivative(x):
        is_flat = not isinstance(x[0], (list, np.ndarray))
        if is_flat:
            x = [x]
        
        y = []
        for row in x:
            y_row = []
            for val in row:
                if val >= 0:
                    y_row.append(1)
                else:
                    y_row.append(0)
            if is_flat:
                y.extend(y_row)
            else:
                y.append(y_row)
        return np.array(y)
    
    def a_sigmoid(x):
        '''
        Sigmoid 1 / (1 + e^(-x))
        '''
        y = 1 / (1 + np.exp(x * -1))
        return y
    
    def a_sigmoid_derivative(x):
        '''
        Source: http://neuralnetworksanddeeplearning.com/chap1.html see sigmoid_prime
        '''
        y = x * (1 - x)
        return y
    
    def a_tanh(x):
        '''
        Tanh y = tanh(x)
        '''
        y = np.tanh(x)
        return y
    
    def a_tanh_derivative(x):
        '''
        y = (1 / tanh(x)) - tanh(x)
        '''
        tanh = np.tanh(x)
        y = (1 / tanh) - tanh
        return y
    
    def a_chillmax(x):
        try:
            x = x - np.max(x, axis=1, keepdims=True)
        except Exception as e:
            import ipdb; ipdb.set_trace()
            a = 0
        y = np.exp(x)
        return y
    
    def a_chillmax_derivative(x):
        # Restrict lowest to just above zeor to prevent -infinity
        #x = np.clip(x, a_min=1e-9, a_max=1+1e-9)
        #y = np.log(x) - x
        x = np.clip(x, 1e-7, 1+1e-7)
        y = np.log(x)
        return y
    
    def a_softmax(x):
        '''
        Softmax e^x with relative normalization and inf protection
        '''
        # limit largest input to zero to protect from overflow
        x = x - np.max(x, axis=1, keepdims=True)
        # Raise euler by inputs; bound range from 0 to 1
        y = np.exp(x)
        # Normalize outputs (confidence percent relative to all other outputs)
        y = y / np.sum(y, axis=1, keepdims=True)
        return y
    
    def a_softmax_derivative(x):
        '''
        TODO: This isn't done. Need to figure out how to extract jacobian values
        Sources:
            * https://automata88.medium.com/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
            * https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
            * https://e2eml.school/softmax.html
        '''
        '''
        is_flat = not isinstance(x[0], (list, np.ndarray))
        if is_flat:
            x = [x]
        
        y = []
        for row in x:
            row_len = len(row)
            jacobian_m = np.zeros((row_len, row_len))

            for i in range(row_len):
                for j in range(row_len):
                    kronecker = 1 if i == j else 0
                    jacobian_m[i][j] = row[i] * (kronecker -  row[j])
            if is_flat:
                y.extend((jacobian_m[0][0], jacobian_m[0][0]))
            else:
                y.append((jacobian_m[0][0], jacobian_m[0][0]))
        
        return np.array(y)
        '''
        return np.ones(x.shape) * 0.000001 # Yeah don't ask
        # (Michael Glazunov - https://stackoverflow.com/questions/57631507/how-can-i-take-the-derivative-of-the-softmax-output-in-back-prop)
        # Mentioned the derivative is jacobian, but when applied it error collapses to aj - tj
        # So I figured if that was the case, return a matrix of ones with same shape as x.
        # That exploded everything to infinity.  During debug I wanted to see why, so I 
        # reduced the matrix values to near zero.
        # Cost started falling so I'm keeping it for now...
        # Speaking of for now, it's either this or ones with a very very small learning rate


class Network(object):
    
    def __init__(self, n_inputs=2, layer_config=[4, 4], n_outputs=1, output_activation='sigmoid', hidden_activation='relu'):
        self.n_inputs = n_inputs
        self.layer_config = layer_config
        self.hidden_layers = []
        self.output_layer = None
        
        for size in layer_config:
            new_layer = Layer(n_inputs=n_inputs, n_neurons=size, activation=hidden_activation)
            self.hidden_layers.append(new_layer)
            n_inputs = size
        
        if n_outputs:
            if n_outputs == 1 and output_activation == 'softmax':
                raise Exception("Cannot used softmax activation with single output neuron: use sigmoid or relu instead")
            self.output_layer = Layer(n_inputs=n_inputs, n_neurons=n_outputs, activation=output_activation)
    
    def run(self, inputs=[]):
        self.inputs = inputs
        self.outputs_hidden = []
        
        outputs = self.inputs
        for hidden_layer in self.hidden_layers:
            outputs = hidden_layer.fire(outputs)
            self.outputs_hidden.append(outputs)
        self.outputs = self.output_layer.fire(outputs)
        return self.outputs
    
    def cost(self, desired):
        '''
        Root mean squared error
        '''
        n_samples = len(desired)
        if n_samples != len(self.outputs):
            raise Exception("Cannot get cost with mismatched outputs: %d outputs to %d desired" % (len(self.outputs, n_samples)))
        
        cost = np.sum(np.square(desired - self.outputs))
        cost /= n_samples
        return cost
    
    def train(self, inputs, outputs, size=50, limit=None, epochs=10, l_rate=1.0, l_momentum=0.0, show_cost=True, noise_fn=None, noise_chance=0.1):
        '''
        MB-GD (mini-batch gradient descent) if size is greater than 1 (recommended)
        SGD (stochastic gradient descent) if size is 1
        sources:
            * http://neuralnetworksanddeeplearning.com/chap2.html
            * https://ml4a.github.io/ml4a/how_neural_networks_are_trained/
            * https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
        
        inputs - complete inputs for network to train with algined with outputs
        outputs - complete desired outputs of the network from inputs
        size - number of mini-batches (random sets from inputs and outputs) used during an epoch of training
        epochs - number of complete training runs to execute
        l_rate - hyperparam learning rate used to control change amount (smaller - slow but precise | larger - fast but may overshoot)
        l_momentum - hyperparam learning momentum used to include (0.0 to 1.0) previous changes in addition to l_rate (keeps the ball rolling so to speak)
        show_cost - show average mini_batch cost after each gradient descent
        '''
        print_dot_per_runs = 10
        n_samples = len(inputs)
        runs_per_epoch = int(n_samples / size)
        self.previous_adjustments = {}
        
        if limit and runs_per_epoch > limit:
            runs_per_epoch = limit
        
        if n_samples < size:
            raise Exception("Cannot train with batch size %d larger than training samples size %d" % (size, n_samples))
        
        print("Training Starting...")
        print("Epochs: %d" % epochs)
        print("Training Data Size: %d" % n_samples)
        print("Mini Batch Size: %d" % size)
        print("Runs per Epoch: %d" % runs_per_epoch)
        print("Learning Rate: %f" % l_rate)
        print("Learning Momentum: %.2f%%" % (l_momentum * 100))
        
        for epoch in range(epochs):
            print("Epoch %d of %d (running %d gradient descents)" % (epoch + 1, epochs, runs_per_epoch))
            mini_batches = get_mini_batch(inputs, outputs, size=size, add_noise=noise_fn, noise_chance=noise_chance)
            sys.stdout.write("\tRuns: ")
            
            cost = 0
            for run in range(runs_per_epoch):
                # Unpack mini_batch
                mini_batch = mini_batches[run]
                mini_batch_inputs = []
                mini_batch_outputs = []
                for x, y in mini_batch:
                    mini_batch_inputs.append(x)
                    mini_batch_outputs.append(y)
                
                # Call gradient descent for each mini_batch
                self.gradient_descent(mini_batch_inputs, mini_batch_outputs, l_rate, l_momentum)
                
                if show_cost:
                    cost += self.cost(mini_batch_outputs)
                    if run == 0:
                        sys.stdout.write("%.2f" % cost)
                
                if run % print_dot_per_runs == 0:
                    sys.stdout.write(".")
                    sys.stdout.flush()
            print()
            if show_cost:
                print("\tAvg Mini-Batch Cost: %.2f" % (cost / runs_per_epoch))
        
        print("Training Complete")
    
    def gradient_descent(self, inputs, expected_outputs, l_rate, l_momentum):
        '''
        inputs - mini_batch input set for network algined with expected_outputs
        expected_outputs - mini_batch desired outputs of the network from inputs
        l_rate - see train method
        l_momentum - see train method
        '''
        # Forward pass
        self.run(inputs)
        
        # Backward pass
        layer = self.output_layer
        error = layer.outputs - expected_outputs
        
        #import ipdb; ipdb.set_trace()
        
        delta = self.backprop(layer, error, l_rate, l_momentum)
        last_layer = layer
        
        n_hidden_layers = len(self.hidden_layers)
        # Run through hidden layers last to first
        for l_index in reversed(range(n_hidden_layers)):
            layer = self.hidden_layers[l_index]
            # Calculate new error based on previous delta error with respect to
            # previous layer's weights (in backward pass, previous layer is the
            # next layer in the network)
            error = np.dot(delta, last_layer.weights.transpose())
            
            delta = self.backprop(layer, error, l_rate, l_momentum)
            last_layer = layer
    
    def backprop(self, layer, error, l_rate, l_momentum):
        '''
        layer - network layer to apply changes
        error - calculated error between result of the layer and desired outputs
        l_rate - see train method
        l_momentum - see train method
        '''
        n_samples = len(layer.outputs)
        # Find gradients from derivative of activation function using layer outputs
        d_gradient = layer.backward_fn(layer.outputs)
        delta = error * d_gradient
        
        # Sum delta on axis 0 to align with layer's biases
        delta_per_bias = np.sum(delta, axis=0)
        
        # Adjust weights in proportion to activation outputs from last layer (this layer's inputs)
        #sum_inputs = np.sum(layer.inputs, axis=0)
        #delta_per_weight = []
        #for s_index in range(n_samples):
        #    inputs = layer.inputs[s_index]
        #    delta_per_weight.append(delta * inputs)
        delta_per_weight = np.dot(layer.inputs.transpose(), delta) #delta * layer.inputs
        # Transpose and sum on axis 1 to align with layer's weights
        #delta_per_weight = np.sum(delta_per_weight.transpose(), axis=1, keepdims=True)
        
        # Get previous adjustments
        prev_adjs = self.previous_adjustments.get(layer, None)
        prev_delta_per_weight = None
        prev_delta_per_bias = None
        if prev_adjs is not None:
            prev_delta_per_weight, prev_delta_per_bias = prev_adjs
        
        # Apply wieghts and biases delta adjustments with optional momentum
        actual_delta_per_weight = l_rate * delta_per_weight
        actual_detla_per_bias = l_rate * delta_per_bias
        
        if prev_delta_per_weight is not None and prev_delta_per_bias is not None:
            actual_delta_per_weight += l_momentum * prev_delta_per_weight
            actual_detla_per_bias += l_momentum * prev_delta_per_bias
        
        layer.weights -= actual_delta_per_weight
        layer.biases -= actual_detla_per_bias
        
        self.previous_adjustments[layer] = (actual_delta_per_weight, actual_detla_per_bias)
        
        return delta
    
    def copy(self):
        new_network = Network(n_inputs=0, layer_config=[], n_outputs=0)
        new_network.n_inputs = self.n_inputs
        new_network.layer_config = self.layer_config.copy()
        for layer in self.hidden_layers:
            new_network.hidden_layers.append(layer.copy())
        new_network.output_layer = self.output_layer.copy()
        return new_network
    
    def fromjson(self, json_data):
        self.n_inputs = json_data['n_inputs']
        self.layer_config = json_data['layer_config']
        
        for layer_data in json_data['hidden_layers']:
            layer = Layer(n_inputs=0, n_neurons=0)
            layer.fromjson(layer_data)
            self.hidden_layers.append(layer)
        
        # TODO: Remove default activation; set it to None when files are migrated
        output_layer = Layer(n_inputs=0, n_neurons=0, activation='softmax')
        output_layer.fromjson(json_data['output_layer'])
        self.output_layer = output_layer
    
    def tojson(self):
        json_data = {'hidden_layers': []}
        json_data['n_inputs'] = self.n_inputs
        json_data['layer_config'] = self.layer_config
        
        for layer in self.hidden_layers:
            json_data['hidden_layers'].append(layer.tojson())
        
        json_data['output_layer'] = self.output_layer.tojson()
        return json_data
    
    def load(filename):
        new_network = Network(n_inputs=0, layer_config=[], n_outputs=0)
        with open(filename, 'r') as f:
            json_data = json.load(f)
        new_network.fromjson(json_data)
        return new_network
    
    def save(self, filename):
        json_data = self.tojson()
        with open(filename, 'w') as f:
            json.dump(json_data, f, sort_keys=True, indent=2)


def get_mini_batch(inputs, outputs, size=50, add_noise=None, noise_chance=0.1):
    mini_batches = []
    n_samples = len(inputs)
    n_mini_batches = int(n_samples / size)
    
    if n_samples != len(outputs):
        raise Exception("Inputs (len %d) and Outputs (len %d) should be the same size" % (n_samples, len(outputs)))
    
    indices = list(range(n_samples))
    shuffle(indices)
    
    for mini_batch_index in range(n_mini_batches):
        start = mini_batch_index * size
        end = start + size
        mini_batch = []
        for index in indices[start:end]:
            if add_noise and random() <= noise_chance:
                data = deepcopy(inputs[index])
                add_noise(data)
            else:
                data = inputs[index]
            mini_batch.append((data, outputs[index]))
        mini_batches.append(mini_batch)
    
    return mini_batches


ACTIVATION_LOOKUP = {
    "relu": {
        "forward": Layer.a_relu,
        "backward": Layer.a_relu_derivative
    },
    "sigmoid": {
        "forward": Layer.a_sigmoid,
        "backward": Layer.a_sigmoid_derivative
    },
    "tanh" : {
        "forward": Layer.a_tanh,
        "backward": Layer.a_tanh_derivative
    },
    "chillmax": {
        "forward": Layer.a_chillmax,
        "backward": Layer.a_chillmax_derivative
    },
    "softmax": {
        "forward": Layer.a_softmax,
        "backward": Layer.a_softmax_derivative
    }
}
