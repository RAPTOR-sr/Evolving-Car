# Evolving Car Simulation

A genetic algorithm project where cars learn to navigate uneven terrain through evolution and neural networks.



## Overview

This project simulates a population of cars that learn to traverse uneven terrain through artificial evolution. Each car is controlled by a simple neural network, and the genetic algorithm selects and breeds the best-performing cars over multiple generations.

## Features

- Physical car simulation using Box2D physics engine
- Neural network controllers that learn to drive on rough terrain
- Genetic algorithm implementation with selection, crossover, and mutation
- Real-time visualization with Pygame

## How It Works

1. **Car Design**: Each car consists of a rectangular body and two wheels connected via revolute joints.
2. **Neural Controller**: Cars are controlled by a simple neural network that takes the car's position, angle, and velocity as inputs and produces wheel motor speeds as outputs.
3. **Fitness Function**: Cars are evaluated based on how far they travel within a set time period.
4. **Evolution**: The best-performing cars are selected to reproduce, creating offspring with mixed traits from their "parents" plus some random mutations.
5. **Generations**: This process repeats over multiple generations, with each new population showing improved performance.

## Requirements

- Python 3.6+
- Pygame
- Box2D
- NumPy

## Installation

```bash
# Clone this repository
git clone https://github.com/RAPTOR-sr/evolving-car.git
cd evolving-car

# Install dependencies
pip install pygame box2d numpy
```

## Usage

Simply run the main Python file:

```bash
python main.py
```

## Configuration

You can modify these parameters in the `main.py` file:

- `POPULATION_SIZE`: Number of cars per generation
- `GENERATIONS`: Number of generations to run
- `MUTATION_RATE`: Probability of mutation during reproduction
- `SIMULATION_TIME`: How long each car is allowed to run

## Extending the Project

Some ideas for extending this project:
- Add more complex terrain features like hills and gaps
- Implement different car designs
- Add more sensors to the neural network input
- Try different neural network architectures
- Add a user interface for adjusting parameters in real-time

## License



## Acknowledgments

- Box2D for the physics simulation
- Pygame for the visualization framework
