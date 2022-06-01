# NeuralNetPy
Neural Net from scratch in Python

## Modules
### Generate
Genetic Algorithm implementation, for those brute force situations

### Neural
Neural Network implementations, including trainer (MB-GD / SGD)


## Trainer Modules
*bp* indicates trainer using backpropagation

*evo* indicates using genetic algorithm

## Training Sets
### MNIST
Collection of handwritten digits from [MNIST](http://yann.lecun.com/exdb/mnist/)

Modules:
* mnist_data: loads MNIST from ubyte files and can draw in terminals
* bmp: a 16-bit BMP file reading module

Includes some user recorded digits BMP files

### Sentences
Generates random word sentences.

Includes a JSON file of pre-generated sentences with labels 0 or 1 if it includes test (1) or not (0)


## Saved Networks
Each saved network directory has *trainers* to indicate where a trained networks accuracy was achived, so it can be reproduced.

### DR Net (dr_net)
Digit Recognition Network (trained on MNIST)
