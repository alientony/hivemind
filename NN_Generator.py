import numpy as np
import random
import math
import copy
import multiprocessing
import pickle
import os
import traceback
import logging
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import random
import math
import copy
import multiprocessing
import pickle
import os
import traceback
import logging
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800
BLOCK_SIZE = 20
POPULATION_SIZE = 1000
GENERATIONS = 3000
MUTATION_RATE = 0.1

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.original_output_size = output_size
        self.extended_output_size = output_size + 10  # Add 10 for movement bot predictions (2 per movement bot)

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, self.extended_output_size))

        self.model = nn.Sequential(*layers)

        # Move the model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)
        return self.model(x)

    def mutate(self, rate):
        with torch.no_grad():
            for param in self.parameters():
                noise = torch.randn_like(param) * rate
                param.add_(noise)
                param.clamp_(-5, 5)  # Clip weights and biases

    def copy_nn(self):
        new_nn = NeuralNetwork(self.input_size, self.hidden_sizes, self.original_output_size)
        new_nn.load_state_dict(self.state_dict())
        return new_nn

    def load_state_dict(self, state_dict):
        if 'model.4.weight' in state_dict and state_dict['model.4.weight'].shape[0] != self.extended_output_size:
            # Extend the last layer's weights and biases
            last_weight = state_dict['model.4.weight']
            last_bias = state_dict['model.4.bias']

            extended_weight = torch.zeros((self.extended_output_size, last_weight.shape[1]), device=last_weight.device)
            extended_weight[:last_weight.shape[0], :] = last_weight
            extended_weight[last_weight.shape[0]:, :] = torch.randn((self.extended_output_size - last_weight.shape[0], last_weight.shape[1]), device=last_weight.device) * 0.01

            extended_bias = torch.zeros(self.extended_output_size, device=last_bias.device)
            extended_bias[:last_bias.shape[0]] = last_bias
            extended_bias[last_bias.shape[0]:] = torch.randn(self.extended_output_size - last_bias.shape[0], device=last_bias.device) * 0.01

            state_dict['model.4.weight'] = extended_weight
            state_dict['model.4.bias'] = extended_bias

        super().load_state_dict(state_dict)

class Bot:
    def __init__(self, x, y, bot_type):
        self.x = x
        self.y = y
        self.type = bot_type
        self.direction = 0
        self.velocity = 0
        self.angular_velocity = 0
        self.max_velocity = 5
        self.max_angular_velocity = math.pi / 16

    def move(self, action):
        if self.type == "sight":
            if action == 0:
                self.angular_velocity = -self.max_angular_velocity
            elif action == 1:
                self.angular_velocity = self.max_angular_velocity
            else:
                self.angular_velocity = 0
        else:
            if action == 0:
                self.velocity = min(self.velocity + 0.5, self.max_velocity)
            elif action == 1:
                self.velocity = max(self.velocity - 0.5, -self.max_velocity)
            elif action == 2:
                self.angular_velocity = -self.max_angular_velocity
            elif action == 3:
                self.angular_velocity = self.max_angular_velocity
            else:
                self.angular_velocity = 0

    def update_position(self):
        self.direction += self.angular_velocity
        self.direction %= 2 * math.pi

        self.x += self.velocity * math.cos(self.direction)
        self.y += self.velocity * math.sin(self.direction)

        self.velocity *= 0.95
        self.angular_velocity *= 0.95

        self.x = max(0, min(self.x, SCREEN_WIDTH - BLOCK_SIZE))
        self.y = max(0, min(self.y, SCREEN_HEIGHT - BLOCK_SIZE))

class Block:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color

class World:
    def __init__(self):
        self.sight_bots = []
        self.movement_bots = []
        self.blocks = []
        self.nn = NeuralNetwork(input_size=35, hidden_sizes=[64, 32], output_size=35)
        self.reset()

    # ... (rest of the class remains the same)

    def reset(self):
        self.sight_bots = [Bot(random.uniform(0, SCREEN_WIDTH - BLOCK_SIZE), random.uniform(0, SCREEN_HEIGHT - BLOCK_SIZE), "sight") for _ in range(5)]
        self.movement_bots = [Bot(random.uniform(0, SCREEN_WIDTH - BLOCK_SIZE), random.uniform(0, SCREEN_HEIGHT - BLOCK_SIZE), "movement") for _ in range(5)]

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
            outputs = self.nn(inputs)

            for i, bot in enumerate(self.sight_bots):
                bot.move(outputs[i*2:(i+1)*2].argmax().item())

            for i, bot in enumerate(self.movement_bots):
                bot.move(outputs[10+i*4:10+(i+1)*4].argmax().item())

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

    def push_block(self, bot, block):
        push_force = bot.velocity
        push_direction = bot.direction

        dx = push_force * math.cos(push_direction)
        dy = push_force * math.sin(push_direction)

        block.x += dx
        block.y += dy

        block.x = max(0, min(block.x, SCREEN_WIDTH - BLOCK_SIZE))
        block.y = max(0, min(block.y, SCREEN_HEIGHT - BLOCK_SIZE))

    def check_collision(self, entity1, entity2):
        return (abs(entity1.x - entity2.x) < BLOCK_SIZE and
                abs(entity1.y - entity2.y) < BLOCK_SIZE)

    def get_inputs(self):
        inputs = []

        for bot in self.sight_bots:
            vision = self.get_bot_vision(bot)
            inputs.extend(vision)

        for bot in self.movement_bots:
            inputs.extend([bot.x / SCREEN_WIDTH, bot.y / SCREEN_HEIGHT])

        return torch.tensor(inputs, dtype=torch.float32)

    def get_bot_vision(self, bot):
        vision = [0] * 5

        x, y = bot.x + BLOCK_SIZE // 2, bot.y + BLOCK_SIZE // 2
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

    def evaluate(self):
        scores = {
            'block_positioning': 0,
            'sight_bot_performance': {
                'blocks_seen': 0,
                'self_position_prediction': 0,
                'block_position_prediction': 0
            },
            'movement_bot_performance': {
                'proximity': 0,
                'push_efficiency': 0
            }
        }
        max_scores = {
            'block_positioning': 0,
            'sight_bot_performance': {
                'blocks_seen': 0,
                'self_position_prediction': 0,
                'block_position_prediction': 0
            },
            'movement_bot_performance': {
                'proximity': 0,
                'push_efficiency': 0
            }
        }
        directional_movement_score = 0
        max_directional_movement_score = 0
        # Define target positions for each color
        targets = [
            (BLUE, 0, 0),
            (GREEN, 0, SCREEN_HEIGHT // 2),
            (RED, 0, SCREEN_HEIGHT - BLOCK_SIZE)
        ]

        # Evaluate block positions
        blocks_touched = set()
        for movement_bot in self.movement_bots:
            for block in self.blocks:
                if self.check_collision(movement_bot, block):
                    blocks_touched.add(block)

        for block in self.blocks:
            if block in blocks_touched:
                target = next((t for t in targets if t[0] == block.color), None)
                if target:
                    horizontal_score = 1 - (block.x / SCREEN_WIDTH)
                    vertical_distance = abs(block.y - target[2])
                    vertical_score = 1 - (vertical_distance / SCREEN_HEIGHT)
                    proximity_score = 1 / (1 + 0.01 * (block.x**2 + (block.y - target[2])**2))
                    right_side_penalty = max(0, (block.x - SCREEN_WIDTH/2) / (SCREEN_WIDTH/2))

                    scores['block_positioning'] += horizontal_score * 2 + vertical_score + proximity_score * 3 - right_side_penalty
                    max_scores['block_positioning'] += 6  # 2 + 1 + 3
            else:
                # Penalty for untouched blocks
                scores['block_positioning'] -= 1
                max_scores['block_positioning'] += 6  # Same max score as touched blocks

        # Evaluate sight bots
        for sight_bot in self.sight_bots:
            vision = self.get_bot_vision(sight_bot)
            blocks_seen = sum(vision[:3])
            scores['sight_bot_performance']['blocks_seen'] += blocks_seen / 3
            max_scores['sight_bot_performance']['blocks_seen'] += 1

            inputs = self.get_inputs()
            outputs = self.nn(inputs)

            # Self position prediction
            self_position_prediction = outputs[-15:-13]
            actual_position = torch.tensor([sight_bot.x / SCREEN_WIDTH, sight_bot.y / SCREEN_HEIGHT], device=outputs.device)
            self_position_error = torch.mean(torch.abs(self_position_prediction - actual_position)).item()
            self_position_score = 1 - self_position_error
            scores['sight_bot_performance']['self_position_prediction'] += self_position_score * 2
            max_scores['sight_bot_performance']['self_position_prediction'] += 2

            # Block position prediction
            if blocks_seen > 0:
                seen_block = next((block for block in self.blocks if self.check_collision(sight_bot, block)), None)
                if seen_block:
                    block_position_prediction = outputs[-13:-11]
                    actual_block_position = torch.tensor([seen_block.x / SCREEN_WIDTH, seen_block.y / SCREEN_HEIGHT], device=outputs.device)
                    block_position_error = torch.mean(torch.abs(block_position_prediction - actual_block_position)).item()
                    block_position_score = 1 - block_position_error
                    scores['sight_bot_performance']['block_position_prediction'] += block_position_score * 2
                    max_scores['sight_bot_performance']['block_position_prediction'] += 2

            # Movement bot position prediction
            movement_bot_predictions = outputs[-11:-1].reshape(5, 2)
            movement_bot_scores = []
            for i, movement_bot in enumerate(self.movement_bots):
                relative_position = torch.tensor([
                    (movement_bot.x - sight_bot.x) / SCREEN_WIDTH,
                    (movement_bot.y - sight_bot.y) / SCREEN_HEIGHT
                ], device=outputs.device)
                prediction_error = torch.mean(torch.abs(movement_bot_predictions[i] - relative_position)).item()
                movement_bot_scores.append(1 - prediction_error)

            avg_movement_bot_score = sum(movement_bot_scores) / len(movement_bot_scores) if movement_bot_scores else 0
            scores['sight_bot_performance']['movement_bot_prediction'] = avg_movement_bot_score * 5
            max_scores['sight_bot_performance']['movement_bot_prediction'] = 5


        for movement_bot in self.movement_bots:
            previous_position = (movement_bot.x - movement_bot.velocity * math.cos(movement_bot.direction),
                                movement_bot.y - movement_bot.velocity * math.sin(movement_bot.direction))

            for block in self.blocks:
                previous_distance = self.distance_between_points(previous_position, (block.x, block.y))
                current_distance = self.distance(movement_bot, block)

                if current_distance < previous_distance:
                    # Calculate the angle between the bot's movement direction and the direction to the block
                    direction_to_block = math.atan2(block.y - movement_bot.y, block.x - movement_bot.x)
                    angle_difference = abs(movement_bot.direction - direction_to_block)
                    angle_difference = min(angle_difference, 2 * math.pi - angle_difference)

                    # Score based on how closely the bot is moving towards the block
                    if angle_difference < math.pi / 2:  # Only count if moving generally towards the block
                        directional_score = (1 - angle_difference / (math.pi / 2)) * (previous_distance - current_distance)
                        directional_movement_score += directional_score

                max_directional_movement_score += self.distance(movement_bot, block)  # Maximum possible improvement

        # Normalize the directional movement score
        if max_directional_movement_score > 0:
            normalized_directional_movement_score = directional_movement_score / max_directional_movement_score
        else:
            normalized_directional_movement_score = 0

        # Add the new metric to the scores and max_scores dictionaries
        scores['movement_bot_performance']['directional_movement'] = normalized_directional_movement_score * 5
        max_scores['movement_bot_performance']['directional_movement'] = 5


        # Evaluate movement bots
        movement_bot_proximity = []
        blocks_pushed = 0
        for movement_bot in self.movement_bots:
            distances = [self.distance(movement_bot, block) for block in self.blocks]
            closest_distance = min(distances)
            movement_bot_proximity.append(1 / (1 + 0.01 * closest_distance))

            for block in self.blocks:
                if self.check_collision(movement_bot, block):
                    blocks_pushed += 1

        avg_proximity = sum(movement_bot_proximity) / len(movement_bot_proximity) if movement_bot_proximity else 0
        push_efficiency = blocks_pushed / (len(self.movement_bots) * len(self.blocks))

        scores['movement_bot_performance']['proximity'] = avg_proximity * 5
        scores['movement_bot_performance']['push_efficiency'] = push_efficiency * 5
        max_scores['movement_bot_performance']['proximity'] = 5
        max_scores['movement_bot_performance']['push_efficiency'] = 5

        # Calculate percentages
        percentages = {
            'block_positioning': (scores['block_positioning'] / max_scores['block_positioning']) * 100 if max_scores['block_positioning'] > 0 else 0,
            'sight_bot_performance': {k: (scores['sight_bot_performance'][k] / max_scores['sight_bot_performance'][k]) * 100
                                      if max_scores['sight_bot_performance'][k] > 0 else 0
                                      for k in scores['sight_bot_performance']},
            'movement_bot_performance': {k: (scores['movement_bot_performance'][k] / max_scores['movement_bot_performance'][k]) * 100
                                        if max_scores['movement_bot_performance'][k] > 0 else 0
                                        for k in scores['movement_bot_performance']}
        }

        # Update the total fitness calculation
        total_score = (scores['block_positioning'] +
                      sum(scores['sight_bot_performance'].values()) +
                      sum(scores['movement_bot_performance'].values()))
        total_max_score = (max_scores['block_positioning'] +
                          sum(max_scores['sight_bot_performance'].values()) +
                          sum(max_scores['movement_bot_performance'].values()))
        total_fitness = total_score / total_max_score if total_max_score > 0 else 0

        return total_fitness, percentages

    def distance_between_points(self, point1, point2):
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

    def distance(self, entity1, entity2):
        return ((entity1.x - entity2.x) ** 2 + (entity1.y - entity2.y) ** 2) ** 0.5

    def distance_between_points(self, point1, point2):
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

    def copy_nn(self):
        return copy.deepcopy(self.nn)

    def set_nn(self, nn):
        self.nn = copy.deepcopy(nn)

def setup_logger(name):
    logger = logging.getLogger(name)

    # Clear any existing handlers
    if logger.handlers:
        logger.handlers = []

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent the logger from propagating messages to the root logger
    logger.propagate = False

    return logger

def run_world(args):
    world_id, generations, result_queue, progress_queue = args
    logger = setup_logger(f"World-{world_id}")
    logger.info(f"Starting world {world_id}")
    try:
        world = World()
        best_fitness = 0
        best_nn = None

        for generation in range(generations_per_run):
            for _ in range(1000):
                world.step()

            fitness = world.evaluate()
            logger.info(f"Generation {generation}, Fitness: {fitness}")

            # Always mutate, but keep the best
            mutated_nn = world.copy_nn()
            mutated_nn.mutate(MUTATION_RATE)
            world.set_nn(mutated_nn)

            if fitness > best_fitness:
                best_fitness = fitness
                best_nn = world.copy_nn()

            if generation % 10 == 0:
                logger.info(f"World {world_id}, Saving NN at Generation {generation}, Fitness: {fitness}")
                filename = f"nn_generation_{generation}_fitness_{fitness:.4f}.pkl"
                with open(filename, "wb") as f:
                    pickle.dump(world.nn, f)

            progress_queue.put((world_id, generation, fitness))

        result_queue.put((world_id, best_fitness, best_nn))
        return world_id, best_fitness, best_nn
    except Exception as e:
        logger.error(f"Error in world {world_id}: {str(e)}")
        traceback.print_exc()
        return world_id, None, None


def main():
    logger = setup_logger("Main")
    logger.info("Main function started")
    num_worlds = 1
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
            progress_queue = manager.Queue()

            for run in range(total_runs):
                logger.info(f"Starting run {run + 1}/{total_runs}")

                with multiprocessing.Pool(processes=num_worlds) as pool:
                    args = [(i, generations_per_run, result_queue, progress_queue) for i in range(num_worlds)]
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

        if overall_best_nn is not None:
            with open(f"best_networks/overall_best_nn_fitness_{overall_best_fitness:.4f}.pkl", "wb") as f:
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

def main_single():
    logger = setup_logger("MainSingle")
    logger.info("Main function (single) started")

    POPULATION_SIZE = 200
    ELITISM_COUNT = 10
    TOURNAMENT_SIZE = 5
    MIN_POPULATION_AFTER_CULLING = POPULATION_SIZE // 2
    EVALUATIONS_PER_NN = 5  # Number of evaluations per neural network

    try:
        world = World()
        population = [world.nn.copy_nn() for _ in range(POPULATION_SIZE)]
        best_fitness = float('-inf')
        best_nn = None
        best_performance = None

        for generation in range(GENERATIONS):
            logger.info(f"Starting Generation {generation}")

            # Evaluate each network in the population
            fitness_scores = []
            for i, nn in enumerate(population):
                total_fitness = 0
                total_percentages = None

                for _ in range(EVALUATIONS_PER_NN):
                    world.reset()  # Reset the world for a new evaluation
                    world.set_nn(nn)
                    for _ in range(1000):
                        world.step()
                    fitness, percentages = world.evaluate()
                    total_fitness += fitness

                    if total_percentages is None:
                        total_percentages = percentages
                    else:
                        for key in total_percentages:
                            if isinstance(total_percentages[key], dict):
                                for subkey in total_percentages[key]:
                                    total_percentages[key][subkey] += percentages[key][subkey]
                            else:
                                total_percentages[key] += percentages[key]

                avg_fitness = total_fitness / EVALUATIONS_PER_NN
                avg_percentages = {k: (v / EVALUATIONS_PER_NN if isinstance(v, (int, float)) else
                                       {sk: sv / EVALUATIONS_PER_NN for sk, sv in v.items()})
                                   for k, v in total_percentages.items()}

                fitness_scores.append(avg_fitness)

                if i % 10 == 0:
                    logger.info(f"Generation {generation}, Network {i}, Avg Fitness: {avg_fitness}")

                if avg_fitness > best_fitness:
                    best_fitness = avg_fitness
                    best_nn = nn.copy_nn()
                    best_performance = avg_percentages
                    logger.info(f"Generation {generation}, New best fitness: {best_fitness}")
                    logger.info("Performance breakdown of new leader:")
                    logger.info(f"Block positioning: {avg_percentages['block_positioning']:.2f}%")
                    logger.info("Sight bot performance:")
                    logger.info(f"  Blocks seen: {avg_percentages['sight_bot_performance']['blocks_seen']:.2f}%")
                    logger.info(f"  Self position prediction: {avg_percentages['sight_bot_performance']['self_position_prediction']:.2f}%")
                    logger.info(f"  Block position prediction: {avg_percentages['sight_bot_performance']['block_position_prediction']:.2f}%")
                    logger.info(f"  Movement bot prediction: {avg_percentages['sight_bot_performance']['movement_bot_prediction']:.2f}%")
                    logger.info("Movement bot performance:")
                    logger.info(f"  Proximity: {avg_percentages['movement_bot_performance']['proximity']:.2f}%")
                    logger.info(f"  Push efficiency: {avg_percentages['movement_bot_performance']['push_efficiency']:.2f}%")
                    logger.info(f"  Directional movement: {avg_percentages['movement_bot_performance']['directional_movement']:.2f}%")


            # Dynamic fitness threshold
            fitness_threshold = np.mean(fitness_scores) - 0.5 * np.std(fitness_scores)

            # Remove underperforming networks
            population_fitness = list(zip(population, fitness_scores))
            initial_population_size = len(population_fitness)
            population_fitness = [pf for pf in population_fitness if pf[1] >= fitness_threshold]
            culled_count = initial_population_size - len(population_fitness)

            logger.info(f"Generation {generation}: Culled {culled_count} networks. {len(population_fitness)} remaining.")

            if culled_count > 0:
                logger.info(f"Fitness range of culled networks: {min(fitness_scores):.4f} to {fitness_threshold:.4f}")

            if len(population_fitness) < MIN_POPULATION_AFTER_CULLING:
                logger.warning(f"Population size ({len(population_fitness)}) below minimum threshold ({MIN_POPULATION_AFTER_CULLING}). Adding random networks.")

            # If too many networks were removed, add random new ones
            while len(population_fitness) < MIN_POPULATION_AFTER_CULLING:
                new_nn = NeuralNetwork(input_size=35, hidden_sizes=[64, 32], output_size=35)
                total_fitness = 0
                for _ in range(EVALUATIONS_PER_NN):
                    world.reset()
                    world.set_nn(new_nn)
                    new_fitness, _ = world.evaluate()
                    total_fitness += new_fitness
                avg_new_fitness = total_fitness / EVALUATIONS_PER_NN
                population_fitness.append((new_nn, avg_new_fitness))

            logger.info(f"Final population size after potential additions: {len(population_fitness)}")

            # Create the next generation
            population_fitness.sort(key=lambda x: x[1], reverse=True)
            next_generation = [pf[0].copy_nn() for pf in population_fitness[:ELITISM_COUNT]]

            # Adaptive mutation rate
            max_fitness = max(fitness_scores)
            min_fitness = min(fitness_scores)
            fitness_range = max_fitness - min_fitness

            while len(next_generation) < POPULATION_SIZE:
                parent1 = selection(population_fitness, TOURNAMENT_SIZE)
                parent2 = selection(population_fitness, TOURNAMENT_SIZE)

                child = improved_crossover(parent1, parent2)

                # Adaptive mutation rate
                total_child_fitness = 0
                for _ in range(EVALUATIONS_PER_NN):
                    world.reset()
                    world.set_nn(child)
                    child_fitness, _ = world.evaluate()
                    total_child_fitness += child_fitness
                avg_child_fitness = total_child_fitness / EVALUATIONS_PER_NN
                child_fitness_normalized = (avg_child_fitness - min_fitness) / fitness_range if fitness_range > 0 else 0.5
                mutation_rate = MUTATION_RATE * (1 - child_fitness_normalized)
                child.mutate(mutation_rate)

                next_generation.append(child)

            population = next_generation

            logger.info(f"Generation {generation} completed. Best fitness: {best_fitness}")

            if generation % 10 == 0:
                logger.info(f"Saving NNs at Generation {generation}")
                best_filename = f"best_nn_generation_{generation}_fitness_{best_fitness:.4f}.pt"
                torch.save(best_nn.state_dict(), best_filename)

        logger.info(f"Run complete. Best fitness: {best_fitness}")
        logger.info("Final performance breakdown of best network:")
        logger.info(f"Block positioning: {best_performance['block_positioning']:.2f}%")
        logger.info(f"Sight bot performance: {best_performance['sight_bot_performance']:.2f}%")
        logger.info(f"Movement bot performance: {best_performance['movement_bot_performance']:.2f}%")
        torch.save(best_nn.state_dict(), "best_network.pt")
        return best_nn

    except Exception as e:
        logger.error(f"An error occurred in the main function (single): {str(e)}")
        logger.error(traceback.format_exc())
        return None
    finally:
        logger.info("Main function (single) completed")

def selection(population_fitness, tournament_size):
    tournament = random.sample(population_fitness, tournament_size)
    return max(tournament, key=lambda x: x[1])[0]

def improved_crossover(nn1, nn2):
    new_nn = nn1.copy_nn()
    for param1, param2, new_param in zip(nn1.parameters(), nn2.parameters(), new_nn.parameters()):
        mask = torch.rand_like(new_param) < 0.5
        new_param.data = torch.where(mask, param1.data, param2.data)

        # Interpolation
        alpha = torch.rand_like(new_param)
        new_param.data = alpha * param1.data + (1 - alpha) * param2.data
    return new_nn

def crossover(nn1, nn2):
    new_nn = nn1.copy_nn()
    for param1, param2, new_param in zip(nn1.parameters(), nn2.parameters(), new_nn.parameters()):
        mask = torch.rand_like(new_param) < 0.5
        new_param.data = torch.where(mask, param1.data, param2.data)
    return new_nn
if __name__ == "__main__":
    # Reset root logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logger = setup_logger("Main")
    logger.info("Starting evolution process...")

    try:
        best_nn = main_single()
        if best_nn is not None:
            logger.info("Evolution completed successfully.")
            # Save the best neural network
            with open("best_network.pkl", "wb") as f:
                pickle.dump(best_nn, f)
            logger.info("Best neural network saved successfully.")
        else:
            logger.warning("Evolution did not produce a valid neural network.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("Program execution completed")
