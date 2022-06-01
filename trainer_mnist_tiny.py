from time import time
from random import random

from twisted.internet import reactor
from twisted.internet.defer import inlineCallbacks

from neural import Network
from generate import Evolve
from TrainingSets.MNIST import mnist_data as md


def check_fitness(output, exp_output):
    score = 0.0
    '''
    Example output for 50/50 selection of 2 and 5 detected
    output = [0, 0, 0.5, 0, 0, 0.5, 0, 0, 0, 0]
    exp_output = <int>[0-9]
    '''
    guess_confidence = output[exp_output]
    return guess_confidence


def modify_network(network):
    for layer in network.hidden_layers:
        modify_network_layer(layer)
    modify_network_layer(network.output_layer)


def modify_network_layer(layer):
    if random() >= 0.99:
        layer.random()
    weight_mutation_type = random()
    bias_mutation_type = random()
    for connection in layer.weights:
        for n_index in range(len(connection)):
            if weight_mutation_type >= 0.9:
                connection[n_index] += (int(random() * 200000) - 100000) * 0.0000001#0.0001
            elif weight_mutation_type >= 0.45:
                connection[n_index] += (int(random() * 20000) - 10000) * 0.0000001#0.0001
            else:
                connection[n_index] += (int(random() * 2000) - 1000) * 0.0000001#0.0001
    for b_index in range(len(layer.biases[0])):
        mutate_type = random()
        if bias_mutation_type >= 0.9:
            layer.biases[0][b_index] += (int(random() * 200000) - 100000) * 0.0000001#0.0001
        elif bias_mutation_type >= 0.45:
            layer.biases[0][b_index] += (int(random() * 20000) - 10000) * 0.0000001#0.0001
        else:
            layer.biases[0][b_index] += (int(random() * 2000) - 1000) * 0.0000001#0.0001


def no_normalize_input(data):
    return data


def bitmap_image_data(images, cut_off=125):
    for image in images:
        for index in range(len(image)):
            val = image[index]
            if val > cut_off:
                image[index] = 1
            else:
                image[index] = 0


def init():
    global generations_per_data_set, images_per_set, labels, images, evo
    generations_per_data_set = 1
    images_per_set = 200

    labels, images = md.load(start=0, end=images_per_set)
    
    seed_network = Network(n_inputs=784, layer_config=[16, 16], n_outputs=10, output_activation='softmax')
    #seed_network = Network.load("SavedNetworks/dr_net/r1.json")
    
    evo = Evolve(
        seed_network,
        modify_network,
        check_fitness,
        no_normalize_input,
        images,
        labels,
        population=500,
        top_percent=0.1,
        name="mnist_trainer_tiny"
    )


@inlineCallbacks
def main():
    global generations_per_data_set, images_per_set, labels, images, evo, start_time
    #data_start = batch_size
    print("\nGeneration 1")
    result = yield evo.run_generation()
    uptime = time() - start_time
    print("Uptime: %0.2f" % uptime)
    print("Best Fitness: %.2f" % (evo.fitness_best * 100))
    print("Avg Fitness: %.2f" % (evo.fitness_avg * 100))
    for index, performance, cost in result[0:3]:
        print("%03d: %.2f (%.2f)" % (index, performance * 100, cost))
    
    while(True):
        survived_networks = evo.get_fittest_networks()
        evo.populate_generation(survived_networks)
        
        if evo.generation % generations_per_data_set == 0:
            start = int(evo.generation / generations_per_data_set) * images_per_set #- int(images_per_set / 2)
            end = start + images_per_set
            labels, images = md.load(start=start, end=end)
            evo.replace_data(images, labels)
        
        result = yield evo.run_generation()
        uptime = time() - start_time
        if evo.generation % generations_per_data_set == 0:
            print("\nGeneration %d (new data set)" % evo.generation)
        else:
            print("\nGeneration %d" % evo.generation)
        print("Uptime: %0.2f" % uptime)
        print("Best Fitness: %.2f" % (evo.fitness_best * 100))
        print("Avg Fitness: %.2f" % (evo.fitness_avg * 100))
        for index, performance, cost in result[0:3]:
            print("%03d: %.2f (%.2f)" % (index, performance * 100, cost))


def reactorInterupt():
    if reactor.running:
        try:
            reactor.stop()
        except:
            pass


def reactorStopped():
    global start_time
    runtime = time() - start_time
    print("Total Runtime: %.2f" % runtime)


def start():
    global start_time
    start_time = time()
    init()
    reactor.addSystemEventTrigger('before', 'shutdown', reactorInterupt)
    reactor.addSystemEventTrigger('after', 'shutdown', reactorStopped)
    reactor.callWhenRunning(main)
    reactor.run()


def test(network="SavedNetworks/dr_net/r1.json", draw=False):
    start_time = time()
    score = 0
    avg_correct_confidence = 0.0
    avg_incorrect_confidence = 0.0
    
    test_network = Network.load(network)
    
    labels, images = md.load(data_set='testing', start=0, end=None)
    small_images = []
    for image in images:
        small_images.append(md.shrink(image))
    
    results = test_network.run(small_images)
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
        
        if draw:
            image = small_images[index]
            md.draw_flat(image, width=14)
            result_status = "WRONG"
            if guess_number == label:
                result_status = "CORRECT"
            print("Label: %d" % label)
            print("Result: %d (%.2f - %s)" % (guess_number, guess_confidence, result_status))
    
    avg_correct_confidence /= test_count
    avg_incorrect_confidence /= test_count
    avg_score = score / test_count * 100
    runtime = time() - start_time
    
    print("Runtime: %.2fs" % runtime)
    print("Accuracy: %.2f%%" % avg_score)
    print("Tests Run: %d" % test_count)
    print("Correct Results: %d (avg confidence %.2f%%)" % (score, avg_correct_confidence))
    print("Incorrect Results: %d (avg confidence %.2f%%)" % ((test_count - score), avg_incorrect_confidence))
