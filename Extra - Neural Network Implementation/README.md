# Simple Neural Network Implementation

This is a simple Python script that serves as a clear and educational introduction to the fundamental functions of a neural network. The implementation focuses on illustrating the concepts of forward and backward passes, weight adjustments, and the sigmoid activation function.

## Purpose

The primary goal is to provide a basic yet effective resource for individuals seeking to gain an intuition about how neural networks operate internally. It is designed as an educational guide to help users understand the foundational aspects of neural networks without the complexities of advanced architectures.

## Script Details

**`sigmoid(x)`**: Implementation of the sigmoid activation function.

**`propagate(x_1, x_2, target, iterations=500, learning_rate=1)`**: 
- Main function for training the neural network through forward and backward passes.
- Initial weights and architecture are defined within the function.
- Prints detailed information about each iteration, including forward and backward passes.

## Usage

Run the script:

```bash
python3 neural_network.py
```

Enter the values for x1, x2, and the target value when prompted.

Observe the forward and backward passes, including weight adjustments, during each iteration.

The script will terminate when the target value is reached or after the specified number of iterations.
