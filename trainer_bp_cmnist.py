from time import time
from random import random
from copy import deepcopy

import numpy as np

from neural import Network
from TrainingSets.MNIST import mnist_data as md


class Optical_CNNetwork(object):  
    def init(self, width, height, pixels, perceptrons, output):
        self.width = width
        self.height = height
        self.pixels = pixels
        self.networks = []
        self.output_network = output.copy()
        
        for y in range(self.height):
            network_row = []
            for x in range(self.width):
                network_row.append(perceptrons.copy())
            self.networks.append(network_row)
    
    def run(self, inputs):
        is_flat = not isinstance(inputs[0], (list, np.ndarray))
        if is_flat:
            inputs = [inputs]
        
        input_height = int(len(inputs[0]) / (self.width * self.pixels))
        if input_height != self.height:
            raise Exception("Image dimension missmatch")
        
        perc_network_inputs = []
        for y in range(self.height):
            for x in range(self.width):
                network_inputs = []
                for data in inputs:
                    data[
            


def noise_generator(data, noise=0.1):
    '''
    left = 0.0
    right = 0.0
    top = 0.0
    bottom = 0.0
    if random() >= 0.5:
        left = 0.5
    else:
        right = 0.5
    if random() >= 0.5:
        top = 0.5
    else:
        bottom = 0.5
    md.align_flat(data, left=left, right=right, top=top, bottom=bottom)
    '''
    for i in range(len(data)):
        if random() <= noise:
            data[i] = int(random() * 255)


def vectorize_onehot_labels(labels):
    vector_labels = []
    for label_index in labels:
        new_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        new_vector[label_index] = 1
        vector_labels.append(new_vector)
    return vector_labels


def init():
    global epochs, learning_rate, learning_momentum, n_mini_batches, noise_chance, labels, inputs, network
    epochs = 80
    learning_rate = 0.1
    learning_momentum = 0.5
    n_mini_batches = 100
    noise_chance = 0.0
    
    print("Loading Training Data...")
    labels, images = md.load(start=0, end=None)
    inputs = images
    
    original_len = len(labels)
    print("Extending Training Data Left...")
    for index in range(original_len):
        label = labels[index]
        image = deepcopy(images[index])
        md.align_flat(image, left=1, relative=False)
        labels.append(label)
        images.append(image)
    print("Extending Training Data Right...")
    for index in range(original_len):
        label = labels[index]
        image = deepcopy(images[index])
        md.align_flat(image, right=1, relative=False)
        labels.append(label)
        images.append(image)
    print("Extending Training Data Top...")
    for index in range(original_len):
        label = labels[index]
        image = deepcopy(images[index])
        md.align_flat(image, top=1, relative=False)
        labels.append(label)
        images.append(image)
    print("Extending Training Data Bottom...")
    for index in range(original_len):
        label = labels[index]
        image = deepcopy(images[index])
        md.align_flat(image, bottom=1, relative=False)
        labels.append(label)
        images.append(image)
    print("Extending Training Data Left-Top...")
    for index in range(original_len):
        label = labels[index]
        image = deepcopy(images[index])
        md.align_flat(image, left=1, top=1, relative=False)
        labels.append(label)
        images.append(image)
    print("Extending Training Data Left-Bottom...")
    for index in range(original_len):
        label = labels[index]
        image = deepcopy(images[index])
        md.align_flat(image, left=1, bottom=1, relative=False)
        labels.append(label)
        images.append(image)
    print("Extending Training Data Right-Top...")
    for index in range(original_len):
        label = labels[index]
        image = deepcopy(images[index])
        md.align_flat(image, right=1, top=1, relative=False)
        labels.append(label)
        images.append(image)
    print("Extending Training Data Right-Bottom...")
    for index in range(original_len):
        label = labels[index]
        image = deepcopy(images[index])
        md.align_flat(image, right=1, bottom=1, relative=False)
        labels.append(label)
        images.append(image)
    #inputs = np.clip(images, 0, 1)
    '''
    print("Shrinking images...")
    inputs = []
    for image in images:
        inputs.append(md.shrink(image))
    '''
    
    print("Vectorizing labels...")
    labels = vectorize_onehot_labels(labels)
    
    print("Creating Network...")
    networks = []
    for i in range(7):
        network_row = []
        for j in range(7):
            
    network = Network(n_inputs=784, layer_config=[100, 50, 25], n_outputs=10, output_activation='softmax')
    #network = Network(n_inputs=784, layer_config=[100, 50, 25], n_outputs=10, output_activation='sigmoid')
    #network = Network.load("SavedNetworks/dr_net/trained_r2_91.json")


def main():
    global epochs, learning_rate, learning_momentum, n_mini_batches, noise_chance, labels, inputs, start_time, network

    print("Testing...")
    test(network=network)
    
    network.train(inputs, labels, size=n_mini_batches, limit=None, epochs=epochs,
                  l_rate=learning_rate, l_momentum=learning_momentum,
                  noise_fn=noise_generator, noise_chance=noise_chance)
    
    print("Testing...")
    test(network=network)


def stop():
    global start_time
    runtime = time() - start_time
    print("Total Runtime: %.2f" % runtime)


def start():
    global start_time
    start_time = time()
    init()
    main()
    stop()


def test(network="SavedNetworks/dr_net/r1.json", data_set='testing', draw=False, draw_wrong=False):
    start_time = time()
    score = 0
    avg_correct_confidence = 0.0
    avg_incorrect_confidence = 0.0
    
    test_network = network
    if isinstance(network, str):
        test_network = Network.load(network)
    
    labels, images = md.load(data_set=data_set, start=0, end=None)
    #images = np.clip(images, 0, 1)
    
    vector_labels = vectorize_onehot_labels(labels)
    
    results = test_network.run(images)
    test_count = len(results)
    
    for index in range(test_count):
        result = results[index].tolist()
        label = labels[index]
        
        result = [round(val * 100, 2) for val in result]
        
        guess_confidence = max(result)
        guess_number = result.index(guess_confidence)
        if guess_number == label:
            score += 1
            avg_correct_confidence += guess_confidence
        else:
            avg_incorrect_confidence += guess_confidence
        
        if draw or (draw_wrong and guess_number != label):
            image = images[index]
            md.draw_flat(image, width=28)
            result_status = "WRONG"
            if guess_number == label:
                result_status = "CORRECT"
            print("Label: %d" % label)
            print("Result: %d (%.2f - %s)" % (guess_number, guess_confidence, result_status))
    
    avg_correct_confidence /= test_count
    avg_incorrect_confidence /= test_count
    avg_score = score / test_count * 100
    cost = test_network.cost(vector_labels)
    runtime = time() - start_time
    
    print("Runtime: %.2fs" % runtime)
    print("Cost: %.2f" % cost)
    print("Accuracy: %.2f%%" % avg_score)
    print("Tests Run: %d" % test_count)
    print("Correct Results: %d (avg confidence %.2f%%)" % (score, avg_correct_confidence))
    print("Incorrect Results: %d (avg confidence %.2f%%)" % ((test_count - score), avg_incorrect_confidence))
