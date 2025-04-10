import numpy as np
import random
import time
from Digital_twin import DigitalTwin

class InvertedPendulumGA:
    def __init__(self, population_size, num_actions, simulation_duration, action_resolution, simulation_delta_t):
        self.digital_twin = DigitalTwin()
        self.population_size = population_size
        self.parent_pool_size = 2 #parent_pool_size
        self.num_actions = num_actions
        self.simulation_duration = simulation_duration
        self.action_resolution = action_resolution
        self.simulation_delta_t = simulation_delta_t
        self.simulation_steps = simulation_duration/simulation_delta_t
        self.num_steps = int(simulation_duration / action_resolution)
        self.step_resolution = int(action_resolution / simulation_delta_t)
        self.population = [self.create_individual() for _ in range(population_size)]
        
        fitness_scores = self.evaluate_population()
        print(fitness_scores, "at start")

    def create_individual(self):
        """Create an individual sequence of actions with balanced left and right actions and boundary constraints."""
        actions = np.zeros(self.num_steps, dtype=int)  # Start with neutral actions
        # Initialize a variable to track the net movement direction and magnitude
        net_movement = 0  # Positive for right, negative for left
        
        for i in range(self.num_steps):
            if abs(net_movement) < 100:
                # If net movement is within acceptable bounds, choose any action
                action = np.random.randint(1, self.num_actions)
                # Update net movement based on the chosen action
                if action in [1, 2, 3, 4]:  # Left actions
                    net_movement -= self.digital_twin.action_map[action][1]
                else:  # Right actions
                    net_movement += self.digital_twin.action_map[action-4][1]
            elif net_movement >= 100:
                # If net movement is too far right, choose a left action to balance
                action = np.random.choice([1, 2, 3, 4])
                net_movement -= self.digital_twin.action_map[action][1]
            else:  # net_movement <= -150
                # If net movement is too far left, choose a right action to balance
                action = np.random.choice([5, 6, 7, 8])
                net_movement += self.digital_twin.action_map[action-4][1]

            actions[i] = action
        
        #print(actions)
        return actions


    def simulate(self, actions):
        """Simulate the inverted pendulum with the given actions and return a fitness score."""
        self.digital_twin.theta = 0.
        self.digital_twin.theta_dot = 0.
        self.digital_twin.x_pivot = 0.
        self.digital_twin.steps = 0.
        max_score = 0.

        action_list = actions.tolist()
        while self.digital_twin.steps < self.simulation_steps:
            if self.digital_twin.steps%self.step_resolution == 0 and len(action_list) > 0:
                action = action_list.pop(0)
                direction, duration = self.digital_twin.action_map[action]
                self.digital_twin.perform_action(direction, duration)
            theta, theta_dot, x_pivot, _ = self.digital_twin.step()
            if abs(theta) > max_score:
                max_score = abs(theta)
            if abs(self.digital_twin.x_pivot) > 99:
                #print('not good')
                return -100
        return max_score

    def evaluate_population(self):
        """Evaluate the fitness of the entire population."""
        fitness_scores = [self.simulate(individual) for individual in self.population]
        return fitness_scores

    def select_parents(self, fitness_scores):
        """Select a pool of parent individuals based on their fitness scores."""
        pool_size = min(self.parent_pool_size, len(fitness_scores))
        # Select indices of the top performers to form the pool
        top_performers_indices = np.argsort(fitness_scores)[-pool_size:]
        return [self.population[i] for i in top_performers_indices]


    def crossover(self, parent1, parent2):
        """Perform crossover between two parents to produce an offspring."""
        crossover_point = random.randint(1, self.num_steps - 1)
        offspring = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        return offspring

    def mutate(self, individual, mutation_rate=0.2):
        """Mutate an individual's actions with a given mutation rate."""
        for i in range(self.num_steps):
            if random.random() < mutation_rate:
                individual[i] = random.randint(0, self.num_actions - 1)
        return individual

    def run_generation(self):
        """Run a single generation of the genetic algorithm, using all parents in the pool to create offspring."""
        fitness_scores = self.evaluate_population()
        parents_pool = self.select_parents(fitness_scores)
        
        # Shuffle the parents pool to randomize pairings
        np.random.shuffle(parents_pool)
        
        new_population = []
        while len(new_population) < self.population_size:
            for i in range(0, len(parents_pool), 2):
                # Break the loop if the new population is already filled
                if len(new_population) >= self.population_size:
                    break
                
                # Ensure there's a pair to process
                if i + 1 < len(parents_pool):
                    parent1 = parents_pool[i]
                    parent2 = parents_pool[i + 1]
                    offspring1 = self.crossover(parent1, parent2)
                    offspring2 = self.crossover(parent2, parent1)  # Optional: create a second offspring by reversing the parents
                    
                    # Mutate and add the new offspring to the new population
                    new_population.append(self.mutate(offspring1))
                    if len(new_population) < self.population_size:
                        new_population.append(self.mutate(offspring2))
                    
                    # If the end of the parent pool is reached but more offspring are needed, reshuffle and continue
                    if i + 2 >= len(parents_pool) and len(new_population) < self.population_size:
                        np.random.shuffle(parents_pool)

        # Replace the old population with the new one
        self.population = new_population[:self.population_size]

    def optimize(self, num_generations, fitness_threshold):
        """Optimize the inverted pendulum control over a number of generations or until an individual meets the fitness threshold."""
        for i in range(num_generations):
            self.run_generation()
            # Evaluate the population after this generation
            fitness_scores = self.evaluate_population()
            best_index = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_index]
            
            print(f"Generation: {i}, Best Fitness: {best_fitness}")
            
            # Check if the best individual meets the fitness threshold
            if best_fitness >= fitness_threshold:
                print(f"Stopping early: Individual found with fitness {best_fitness} meeting the threshold at generation {i}.")
                return self.population[best_index]
        
        # If the loop completes without returning, no individual met the threshold; return the best found
        print(f"No individual met the fitness threshold. Best fitness after {num_generations} generations is {best_fitness}.")
        return self.population[best_index]


# Example usage
ga = InvertedPendulumGA(population_size=40, num_actions=9, simulation_duration=2, action_resolution=0.2, simulation_delta_t=0.005)
best_solution = ga.optimize(num_generations=100, fitness_threshold=np.pi)

print("Best Solution:", best_solution)

