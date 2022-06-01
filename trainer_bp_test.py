from time import time

from neural import Network
from TrainingSets.MNIST import mnist_data as md


def init():
    global epochs, learning_rate, n_mini_batches, labels, inputs, network
    epochs = 100
    learning_rate = 0.5
    #n_mini_batches = 10
    n_mini_batches = 5
    
    print("Loading Training Data...")
    training_data = [
        [[2.7810836,2.550537003],[0]],
        [[1.465489372,2.362125076],[0]],
        [[3.396561688,4.400293529],[0]],
        [[1.38807019,1.850220317],[0]],
        [[3.06407232,3.005305973],[0]],
        [[7.627531214,2.759262235],[1]],
        [[5.332441248,2.088626775],[1]],
        [[6.922596716,1.77106367],[1]],
        [[8.675418651,-0.242068655],[1]],
        [[7.673756466,3.508563011],[1]]
    ]
    labels = []
    inputs = []
    for data, label in training_data:
        labels.append(label)
        inputs.append(data)
    
    print("Creating Network...")
    network = Network(n_inputs=2, layer_config=[2], n_outputs=1, output_activation='sigmoid', hidden_activation='sigmoid')
    #network = Network.load("SavedNetworks/test_backprop/00_pretrain.json")


def main():
    global epochs, learning_rate, n_mini_batches, labels, inputs, start_time, network
    
    print("Testing...")
    test(network=network)
    
    network.train(labels, inputs, size=n_mini_batches,
                  epochs=epochs, learning_rate=learning_rate)
    
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


def test(network="SavedNetworks/dr_net/r1.json"):
    global labels, inputs
    start_time = time()
    score = 0
    avg_correct_confidence = 0.0
    avg_incorrect_confidence = 0.0
    
    test_network = network
    if isinstance(network, str):
        test_network = Network.load(network)
    
    results = test_network.run(inputs)
    print(results)
    print(labels)
    test_count = len(results)
    
    cost = test_network.cost(labels)
    runtime = time() - start_time
    
    print("Runtime: %.2fs" % runtime)
    print("Cost: %.2f" % cost)
