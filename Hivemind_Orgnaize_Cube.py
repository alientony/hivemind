import pygame
import numpy as np
import random
from typing import List, Tuple
import math
import copy
import multiprocessing
import pickle
import os
import traceback
import logging

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800
BLOCK_SIZE = 20
POPULATION_SIZE = 100
GENERATIONS = 10000
MUTATION_RATE = 0.3

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

class NeuralNetwork:
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # Initialize weights
        self.weights = []
        self.biases = []

        # Input to first hidden layer
        self.weights.append(np.random.randn(self.input_size, self.hidden_sizes[0]))
        self.biases.append(np.random.randn(self.hidden_sizes[0]))

        # Hidden layers
        for i in range(len(self.hidden_sizes) - 1):
            self.weights.append(np.random.randn(self.hidden_sizes[i], self.hidden_sizes[i + 1]))
            self.biases.append(np.random.randn(self.hidden_sizes[i + 1]))

        # Last hidden layer to output
        self.weights.append(np.random.randn(self.hidden_sizes[-1], self.output_size))
        self.biases.append(np.random.randn(self.output_size))

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.a = []
        self.z = []

        # Input to first hidden layer
        self.z.append(np.dot(X, self.weights[0]) + self.biases[0])
        self.a.append(self.relu(self.z[0]))

        # Hidden layers
        for i in range(1, len(self.hidden_sizes)):
            self.z.append(np.dot(self.a[-1], self.weights[i]) + self.biases[i])
            self.a.append(self.relu(self.z[-1]))

        # Last hidden layer to output
        self.z.append(np.dot(self.a[-1], self.weights[-1]) + self.biases[-1])
        self.a.append(self.sigmoid(self.z[-1]))

        return self.a[-1]

    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def mutate(self, rate: float):
        for i in range(len(self.weights)):
            self.weights[i] += np.random.randn(*self.weights[i].shape) * rate
            self.biases[i] += np.random.randn(*self.biases[i].shape) * rate


class Bot:
    def __init__(self, x: float, y: float, bot_type: str):
        self.x = x
        self.y = y
        self.type = bot_type
        self.direction = 0  # in radians
        self.velocity = 0
        self.angular_velocity = 0
        self.max_velocity = 5
        self.max_angular_velocity = math.pi / 16  # About 11.25 degrees per step

    def move(self, action: int):
        if self.type == "sight":
            if action == 0:
                self.angular_velocity = -self.max_angular_velocity
            elif action == 1:
                self.angular_velocity = self.max_angular_velocity
            else:
                self.angular_velocity = 0
        else:  # movement bot
            if action == 0:  # Accelerate
                self.velocity = min(self.velocity + 0.5, self.max_velocity)
            elif action == 1:  # Decelerate
                self.velocity = max(self.velocity - 0.5, -self.max_velocity)
            elif action == 2:  # Rotate left
                self.angular_velocity = -self.max_angular_velocity
            elif action == 3:  # Rotate right
                self.angular_velocity = self.max_angular_velocity
            else:
                self.angular_velocity = 0

    def update_position(self):
        # Update direction
        self.direction += self.angular_velocity
        self.direction %= 2 * math.pi

        # Update position based on velocity and direction
        self.x += self.velocity * math.cos(self.direction)
        self.y += self.velocity * math.sin(self.direction)

        # Apply friction
        self.velocity *= 0.95  # Reduce velocity by 5% each step
        self.angular_velocity *= 0.95  # Reduce angular velocity by 5% each step

        # Keep bot within screen bounds
        self.x = max(0, min(self.x, SCREEN_WIDTH - BLOCK_SIZE))
        self.y = max(0, min(self.y, SCREEN_HEIGHT - BLOCK_SIZE))

class Block:
    def __init__(self, x: int, y: int, color: Tuple[int, int, int]):
        self.x = x
        self.y = y
        self.color = color

class World:
    def __init__(self):
        self.sight_bots: List[Bot] = []
        self.movement_bots: List[Bot] = []
        self.blocks: List[Block] = []
        # Define the neural network with input size, list of hidden layer sizes, and output size
        self.nn = NeuralNetwork(input_size=35, hidden_sizes=[64, 32], output_size=30)
        
        self.reset()

    def reset(self):
        self.sight_bots = [Bot(random.uniform(0, SCREEN_WIDTH - BLOCK_SIZE), random.uniform(0, SCREEN_HEIGHT - BLOCK_SIZE), "sight") for _ in range(5)]
        self.movement_bots = [Bot(random.uniform(0, SCREEN_WIDTH - BLOCK_SIZE), random.uniform(0, SCREEN_HEIGHT - BLOCK_SIZE), "movement") for _ in range(5)]
        
        # Spawn blocks in the middle of the map
        middle_start = SCREEN_WIDTH // 4
        middle_end = 3 * SCREEN_WIDTH // 4
        self.blocks = [
            Block(random.uniform(middle_start, middle_end), random.uniform(0, SCREEN_HEIGHT - BLOCK_SIZE), RED),
            Block(random.uniform(middle_start, middle_end), random.uniform(0, SCREEN_HEIGHT - BLOCK_SIZE), GREEN),
            Block(random.uniform(middle_start, middle_end), random.uniform(0, SCREEN_HEIGHT - BLOCK_SIZE), BLUE)
        ]

    def step(self):
        try:
            inputs = self.get_inputs()
            outputs = self.nn.forward(inputs)
            
            for i, bot in enumerate(self.sight_bots):
                bot.move(np.argmax(outputs[i*2:(i+1)*2]))
            
            for i, bot in enumerate(self.movement_bots):
                bot.move(np.argmax(outputs[10+i*4:10+(i+1)*4]))
            
            for bot in self.sight_bots + self.movement_bots:
                bot.update_position()
            
            self.handle_collisions()
        except Exception as e:
            print(f"Error in World.step: {str(e)}")
            traceback.print_exc()

    def handle_collisions(self):
        for bot in self.movement_bots:
            for block in self.blocks:
                if self.check_collision(bot, block):
                    self.push_block(bot, block)

    def push_block(self, bot: Bot, block: Block):
        push_force = bot.velocity
        push_direction = bot.direction

        dx = push_force * math.cos(push_direction)
        dy = push_force * math.sin(push_direction)

        block.x += dx
        block.y += dy

        # Keep block within screen bounds
        block.x = max(0, min(block.x, SCREEN_WIDTH - BLOCK_SIZE))
        block.y = max(0, min(block.y, SCREEN_HEIGHT - BLOCK_SIZE))

    def check_collision(self, entity1, entity2):
        return (abs(entity1.x - entity2.x) < BLOCK_SIZE and
                abs(entity1.y - entity2.y) < BLOCK_SIZE)
        
    def get_inputs(self) -> np.ndarray:
        inputs = []
        
        # Inputs for sight bots
        for bot in self.sight_bots:
            vision = self.get_bot_vision(bot)
            inputs.extend(vision)
        
        # Inputs for movement bots (only position)
        for bot in self.movement_bots:
            inputs.extend([bot.x / SCREEN_WIDTH, bot.y / SCREEN_HEIGHT])
        
        return np.array(inputs)

    def get_bot_vision(self, bot: Bot) -> List[float]:
        vision = [0] * 5  # Red, Green, Blue, Other bot, Empty
        
        x, y = bot.x + BLOCK_SIZE // 2, bot.y + BLOCK_SIZE // 2  # Center of the bot
        dx, dy = math.cos(bot.direction), math.sin(bot.direction)
        
        while 0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT:
            x += dx * BLOCK_SIZE
            y += dy * BLOCK_SIZE
            
            for block in self.blocks:
                if block.x <= x < block.x + BLOCK_SIZE and block.y <= y < block.y + BLOCK_SIZE:
                    if block.color == RED:
                        vision[0] = 1
                    elif block.color == GREEN:
                        vision[1] = 1
                    elif block.color == BLUE:
                        vision[2] = 1
                    return vision
            
            for other_bot in self.sight_bots + self.movement_bots:
                if other_bot.x <= x < other_bot.x + BLOCK_SIZE and other_bot.y <= y < other_bot.y + BLOCK_SIZE:
                    vision[3] = 1
                    return vision
        
        vision[4] = 1
        return vision
    
    def evaluate(self) -> float:
        score = 0
        max_score = 0  # We'll use this to normalize the score at the end
    
        # Define target positions
        targets = [
            (BLUE, 0, 0),  # Blue in top left
            (GREEN, 0, SCREEN_HEIGHT // 2),  # Green in middle left
            (RED, 0, SCREEN_HEIGHT - BLOCK_SIZE)  # Red in bottom left
        ]
    
        for block in self.blocks:
            target = next((t for t in targets if t[0] == block.color), None)
            if target:
                # Reward for horizontal positioning (closer to left wall is better)
                horizontal_score = 1 - (block.x / SCREEN_WIDTH)
                score += horizontal_score
                max_score += 1
    
                # Reward/punish for vertical positioning
                vertical_distance = abs(block.y - target[2])
                vertical_score = 1 - (vertical_distance / SCREEN_HEIGHT)
                score += vertical_score
                max_score += 1
    
                # Extra reward for being very close to target position
                if block.x < BLOCK_SIZE and abs(block.y - target[2]) < BLOCK_SIZE:
                    score += 2
                    max_score += 2
    
            # Punish for being on the right side of the screen
            if block.x > SCREEN_WIDTH // 2:
                score -= 0.5
                max_score += 0.5
    
        # Rewards and punishments for bot behaviors
        for sight_bot in self.sight_bots:
            vision = self.get_bot_vision(sight_bot)
            if any(vision[:3]):  # Reward for seeing a block
                score += 0.2
                max_score += 0.2
            elif vision[3]:  # Small punishment for only seeing other bots
                score -= 0.1
                max_score += 0.1
    
        for movement_bot in self.movement_bots:
            # Reward for being close to a block
            distances = [self.distance(movement_bot, block) for block in self.blocks]
            closest_distance = min(distances)    
            proximity_score = 0.5 * (1 - closest_distance / (SCREEN_WIDTH + SCREEN_HEIGHT))
            score += proximity_score
            max_score += 0.5
    
            # Reward for being to the right of a block (encouraging pushing left)
            for block in self.blocks:
                if movement_bot.x > block.x:
                    score += 0.2
                    max_score += 0.2

            # Punish for being on the right side of the screen
            if movement_bot.x > SCREEN_WIDTH // 2:
                score -= 0.3
                max_score += 0.3
    
        # Normalize score between 0 and 1
        return score / max_score if max_score > 0 else 0

    def distance(self, entity1, entity2):
        return ((entity1.x - entity2.x) ** 2 + (entity1.y - entity2.y) ** 2) ** 0.5

    def copy_nn(self):
        return copy.deepcopy(self.nn)

    def set_nn(self, nn):
        self.nn = copy.deepcopy(nn)

def draw_world(world: World, screen):
    screen.fill(WHITE)
    
    for bot in world.sight_bots + world.movement_bots:
        color = BLACK if bot.type == "sight" else (150, 150, 150)
        pygame.draw.rect(screen, color, (int(bot.x), int(bot.y), BLOCK_SIZE, BLOCK_SIZE))
        
        direction_color = (100, 100, 100)
        end_x = bot.x + BLOCK_SIZE // 2 + math.cos(bot.direction) * BLOCK_SIZE
        end_y = bot.y + BLOCK_SIZE // 2 + math.sin(bot.direction) * BLOCK_SIZE
        pygame.draw.line(screen, direction_color, 
                         (int(bot.x + BLOCK_SIZE // 2), int(bot.y + BLOCK_SIZE // 2)),
                         (int(end_x), int(end_y)), 2)
    
    for block in world.blocks:
        pygame.draw.rect(screen, block.color, (int(block.x), int(block.y), BLOCK_SIZE, BLOCK_SIZE))
    
    pygame.display.flip()

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def run_world(args):
    world_id, generations, result_queue = args
    logger = setup_logger(f"World-{world_id}")
    try:
        world = World()
        best_fitness = 0
        best_nn = None

        for generation in range(generations):
            for _ in range(1000):  # Simulation steps per generation
                world.step()

            fitness = world.evaluate()
            if fitness > best_fitness:
                best_fitness = fitness
                best_nn = world.copy_nn()
                best_nn.mutate(MUTATION_RATE)


            if generation % 100 == 0:
                logger.info(f"Generation {generation}, Fitness: {fitness}")

        result_queue.put((world_id, best_fitness, best_nn))
        return world_id, best_fitness, best_nn
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        traceback.print_exc()
        return world_id, None, None

def visualize_best_performer(best_nn):
    print("Initializing Pygame...")
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Bot Evolution Visualization - Best Performer")
    clock = pygame.time.Clock()

    try:
        world = World()
        world.set_nn(best_nn)
        print("World created and neural network set")

        print("Starting visualization. Close the window to exit.")
        running = True
        step = 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            world.step()
            
            screen.fill(WHITE)
            
            # Draw bots and blocks
            draw_world(world, screen)
            
            pygame.display.flip()
            clock.tick(30)  # Limit to 30 FPS

            step += 1
            if step % 100 == 0:
                print(f"Step {step}, Fitness: {world.evaluate()}")

            if step >= 1000:  # Run for 1000 steps then reset
                step = 0
                world.reset()
                world.set_nn(best_nn)

    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        traceback.print_exc()
    finally:
        pygame.quit()
    print("Visualization complete.")

def main():
    logger = setup_logger("Main")
    logger.info("Main function started")
    num_worlds = 10
    generations_per_run = GENERATIONS
    total_runs = 1

    if not os.path.exists("best_networks"):
        os.makedirs("best_networks")
        logger.info("Created 'best_networks' directory")

    overall_best_fitness = 0
    overall_best_nn = None

    try:
        with multiprocessing.Manager() as manager:
            result_queue = manager.Queue()

            for run in range(total_runs):
                logger.info(f"Starting run {run + 1}/{total_runs}")

                with multiprocessing.Pool(processes=num_worlds) as pool:
                    args = [(i, generations_per_run, result_queue) for i in range(num_worlds)]
                    results = pool.map(run_world, args)

                logger.info("All processes completed, collecting results")
                valid_results = [r for r in results if r[1] is not None]
                
                if valid_results:
                    best_world = max(valid_results, key=lambda x: x[1])
                    logger.info(f"Run {run + 1} complete. Best fitness: {best_world[1]}")
                    
                    if best_world[1] > overall_best_fitness:
                        overall_best_fitness = best_world[1]
                        overall_best_nn = best_world[2]
                        logger.info(f"New overall best fitness: {overall_best_fitness}")
                else:
                    logger.warning(f"Run {run + 1} failed to produce valid results.")

        logger.info("All runs complete.")
        logger.info(f"Overall best fitness: {overall_best_fitness}")

        # Save the overall best neural network
        if overall_best_nn is not None:
            with open("best_networks/overall_best_nn.pkl", "wb") as f:
                pickle.dump(overall_best_nn, f)
            logger.info("Best neural network saved successfully.")
        else:
            logger.warning("No valid neural network found to save.")

        return overall_best_nn

    except Exception as e:
        logger.error(f"An error occurred in the main function: {str(e)}")
        logger.error(traceback.format_exc())
        return None
    finally:
        logger.info("Main function completed")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')  # Use 'spawn' instead of 'fork' on Unix
    logger = setup_logger("Main")
    logger.info("Starting evolution process...")
    
    try:
        best_nn = main()
        
        if best_nn is not None:
            logger.info("Evolution complete. Best neural network found. Press Enter to start visualization...")
            input()  # Wait for user input before starting visualization
            visualize_best_performer(best_nn)
        else:
            logger.error("No valid neural network found. Cannot start visualization.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        logger.error(traceback.format_exc())
    finally:

        logger.info("Program execution completed")
