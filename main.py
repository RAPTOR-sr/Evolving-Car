import pygame
import Box2D
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody)

#Configuration

PPM = 20.0  # pixels per meter
TARGET_FPS = 60
Time_Step = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600

#INIT PYGAME
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Evolving Car - Starter Skeleton")
clock = pygame.time.Clock()

#box2d world
world = world(gravity=(0, -9.81), doSleep=True)


#ground body
ground_body = world.CreateStaticBody(
    position=(0, 1))

ground_box = ground_body.CreatePolygonFixture(box=(50, 1), density=0, friction=0.3)

#car body
car_body = world.CreateDynamicBody(position=(5, 5))
box = car_body.CreatePolygonFixture(box=(2, 1), density=1, friction=0.3)

#car wheels
wheel1 = world.CreateDynamicBody(position=(4, 4))
circle1 = wheel1.CreateCircleFixture(radius=0.4, density=1, friction=0.9)

wheel2 = world.CreateDynamicBody(position=(6, 4))
circle2 = wheel2.CreateCircleFixture(radius=0.4, density=1, friction=0.9)

# Revolute joints (attach wheels to body)
joint1 = world.CreateRevoluteJoint(bodyA=car_body, bodyB=wheel1,
                          anchor=wheel1.position,
                          enableMotor=True,
                          maxMotorTorque=1000,
                          motorSpeed=0)

joint2 = world.CreateRevoluteJoint(bodyA=car_body, bodyB=wheel2,
                          anchor=wheel2.position,
                          enableMotor=True,
                          maxMotorTorque=1000,
                          motorSpeed=0)

# SIMPLE NEURAL NETWORK CONTROLLER

class NeuralController:
    def __init__(self, input_size=4, hidden_size=5, output_size=2):
        self.w1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.random.randn(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.random.randn(output_size)

    def Foward(self, x):
        h = np.tanh(np.dot(x, self.w1) + self.b1)
        out = np.tanh(np.dot(h, self.w2) + self.b2)
        return out # output between -1 and 1
    
Controller = NeuralController()


# DRAWING FUNCTION
# --------------------
def draw_polygon(polygon, body, fixture, color=(0, 0, 255)):
    vertices = [(body.transform * v) * PPM for v in polygon.vertices]
    vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
    pygame.draw.polygon(screen, color, vertices)


def draw_circle(circle, body, fixture, color=(255, 0, 0)):
    position = body.transform * circle.pos * PPM
    position = (position[0], SCREEN_HEIGHT - position[1])
    pygame.draw.circle(screen, color, [int(x) for x in position], int(circle.radius * PPM))


# --------------------
# MAIN LOOP
# --------------------
running = True
while running:
    screen.fill((255, 255, 255))

    # Quit Event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --------------------
    # Dummy Controller (apply constant torque to wheels)
    # --------------------
    for joint in world.joints:
        joint.motorSpeed = -30  # negative = forward, positive = backward

    # --------------------
    # Step Physics
    # --------------------
    world.Step(Time_Step, 10, 10)

    # --------------------
    # Draw Bodies
    # --------------------
    for body in world.bodies:
        for fixture in body.fixtures:
            shape = fixture.shape
            if isinstance(shape, polygonShape):
                draw_polygon(shape, body, fixture)
            elif isinstance(shape, circleShape):
                draw_circle(shape, body, fixture)

    pygame.display.flip()
    clock.tick(TARGET_FPS)

pygame.quit()