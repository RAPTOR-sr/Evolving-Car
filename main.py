import pygame
import Box2D
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody)
import numpy as np
import random

#Configuration

PPM = 20.0  # pixels per meter
TARGET_FPS = 60
Time_Step = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
MUTATION_RATE = 0.1
POPULATION_SIZE = 10     # cars per generation
SIMULATION_TIME = 5      # seconds per car
GENERATIONS = 20

#INIT PYGAME
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Evolving Car - Starter Skeleton")
clock = pygame.time.Clock()



# SIMPLE NEURAL NETWORK CONTROLLER

class NeuralController:
    def __init__(self, input_size=4, hidden_size=5, output_size=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.w1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.random.randn(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.random.randn(output_size)

    def Foward(self, x):
        h = np.tanh(np.dot(x, self.w1) + self.b1)
        out = np.tanh(np.dot(h, self.w2) + self.b2)
        return out # output between -1 and 1
    
    def get_weights(self):
        return [self.w1, self.b1 , self.w2 , self.b2]
    
    def set_weights(self,weights):
        self.w1, self.b1 , self.w2 , self.b2 = weights

    def mutate(self):
        for w in [self.w1, self.b1 , self.w2 , self.b2]:
            mask = np.random.rand(*w.shape) < MUTATION_RATE 
            w += mask * np.random.normal(0,0.5, w.shape)

    @staticmethod
    def crossover(parent1 , parent2):
        child = NeuralController()
        w1 , b1 , w2 , b2 = parent1.get_weights()
        w1b, b1b, w2b, b2b = parent2.get_weights()

        #mix weights (50/50)
        new_wights = [
            np.where(np.random.rand(*w1.shape)< 0.5 , w1 , w1b),
            np.where(np.random.rand(*b1.shape)< 0.5 , b1 , b1b),
            np.where(np.random.rand(*w2.shape)< 0.5 , w2 , w2b),
            np.where(np.random.rand(*b2.shape)< 0.5 , b2 , b2b)
            
        ]
        child.set_weights(new_wights)
        return child


# PHYSICS: CAR FACTORY
# --------------------

def create_world_car(controller):
    
    #box2d world
    world_instance = world(gravity=(0, -9.81), doSleep=True)
    
    # Create uneven terrain as a series of points
    terrain_pints = []
    y = 3.0 # strating height
    step_size = 0.5 # CONTROLS SMOOTHNESS
    for x in range(0,50):
        if x < 5:
            y += random.uniform(-step_size*1.5,step_size*1.5)
        else:
            y += random.uniform(-step_size,step_size)
        terrain_pints.append((x,y))
    ground_body = world_instance.CreateStaticBody(shapes=Box2D.b2ChainShape(vertices=terrain_pints))
    


    
    #car body
    car_body = world_instance.CreateDynamicBody(position=(6, 6))
    box = car_body.CreatePolygonFixture(box=(2, 1), density=1, friction=0.3)
    
    #car wheels
    wheel1 = world_instance.CreateDynamicBody(position=(5, 5))
    circle1 = wheel1.CreateCircleFixture(radius=0.8, density=1, friction=0.9)

    wheel2 = world_instance.CreateDynamicBody(position=(7, 5))
    circle2 = wheel2.CreateCircleFixture(radius=0.8, density=1, friction=0.9)

    # Revolute joints (attach wheels to body)
    joint1 = world_instance.CreateRevoluteJoint(bodyA=car_body, bodyB=wheel1,
                            anchor=wheel1.position,
                            enableMotor=True,
                            maxMotorTorque=1000,
                            motorSpeed=0)

    joint2 = world_instance.CreateRevoluteJoint(bodyA=car_body, bodyB=wheel2,
                            anchor=wheel2.position,
                            enableMotor=True,
                            maxMotorTorque=1000,
                            motorSpeed=0)
    
    return world_instance , car_body , joint1 , joint2 , controller

# FITNESS FUNCTION
# --------------------
def run_simulation(controller, visualize=False):
    w, car, j1, j2, ctrl = create_world_car(controller)
    sim_steps = int(SIMULATION_TIME * TARGET_FPS)

    for step in range(sim_steps):
        state = np.array([
            car.position.x,
            car.angle,
            car.linearVelocity.x,
            car.linearVelocity.y
        ])

        action = ctrl.Foward(state)
        left_speed, right_speed = action * 50
        j1.motorSpeed = left_speed
        j2.motorSpeed = right_speed

        w.Step(Time_Step, 10, 10)

        if visualize:
            screen.fill((255, 255, 255))
            for body in w.bodies:
                for fixture in body.fixtures:
                    shape = fixture.shape
                    if isinstance(shape, polygonShape):
                        draw_polygon(shape, body , fixture)
                    elif isinstance(shape , circleShape):
                        draw_circle(shape,body,fixture)
                    elif isinstance(shape, Box2D.b2ChainShape):
                        draw_chain(shape, body)
                    

            pygame.display.flip()
            clock.tick(TARGET_FPS)

    return car.position.x  # fitness = distance traveled

# DRAWING FUNCTION
# --------------------
def draw_polygon(polygon, body, fixture, color=(0, 0, 255)):
    vertices = [(body.transform * v) * PPM for v in polygon.vertices]
    vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
    pygame.draw.polygon(screen, color, vertices)

def draw_chain(chain, body, color=(0, 128, 0)):
    vertices = [(body.transform * v) * PPM for v in chain.vertices]
    vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
    pygame.draw.lines(screen, color, False, vertices, 3)

def draw_circle(circle, body, fixture, color=(255, 0, 0)):
    position = body.transform * circle.pos * PPM
    position = (position[0], SCREEN_HEIGHT - position[1])
    pygame.draw.circle(screen, color, [int(x) for x in position], int(circle.radius * PPM))

# GENETIC ALGORITHM LOOP
# --------------------
population = [NeuralController() for _ in range(POPULATION_SIZE)]

for gen in range(GENERATIONS):
    print(f"Generation {gen+1}")

    # Evaluate fitness
    fitness_scores = []
    for i, ctrl in enumerate(population):
        fitness = run_simulation(ctrl, visualize=(i == 0))  # visualize first car
        fitness_scores.append((fitness, ctrl))

    # Sort by fitness
    fitness_scores.sort(key=lambda x: x[0], reverse=True)
    best_fitness, best_ctrl = fitness_scores[0]
    print(f"  Best fitness: {best_fitness:.2f}")

    # Selection: keep top 50%
    survivors = [ctrl for _, ctrl in fitness_scores[:POPULATION_SIZE // 2]]

    # Reproduce
    new_population = survivors.copy()
    while len(new_population) < POPULATION_SIZE:
        p1, p2 = np.random.choice(survivors, 2, replace=False)
        child = NeuralController.crossover(p1, p2)
        child.mutate()
        new_population.append(child)

    population = new_population

pygame.quit()

