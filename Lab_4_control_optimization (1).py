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
        actions = np.ones(self.num_steps, dtype=int)  # Start with ones instead of zeros
        # Initialize a variable to track the net movement direction and magnitude
        net_movement = 0  # Positive for right, negative for left
        
        for i in range(self.num_steps):
            if abs(net_movement) < 100:
                # If net movement is within acceptable bounds, choose any action
                action = np.random.randint(1, self.num_actions)
                # Update net movement based on the chosen action
                if action in [1, 2, 3, 4,5,6,7,8,9,10,11,12]:  # Left actions
                    net_movement -= self.digital_twin.action_map[action][1]
                else:  # Right actions
                    net_movement += self.digital_twin.action_map[action-4][1]
            elif net_movement >= 100:
                # If net movement is too far right, choose a left action to balance
                action = np.random.choice([1, 2, 3, 4,5,6,7,8,9,10,11,12])
                net_movement -= self.digital_twin.action_map[action][1]
            else:  # net_movement <= -150
                # If net movement is too far left, choose a right action to balance
                action = np.random.choice([13,14,15,16,17,18,19,20,21,22,23,24])
                net_movement += self.digital_twin.action_map[action-4][1]

            actions[i] = action
        
        return actions


    def simulate(self, actions):
        """Simulate the inverted pendulum with the given actions and return a fitness score."""
        self.digital_twin.theta = 0.
        self.digital_twin.theta_dot = 0.
        self.digital_twin.x_pivot = 0.
        self.digital_twin.steps = 0.
        
        # Initialize fitness components
        stability_score = 0.0
        max_deviation = 0.0
        stable_time = 0.0
        total_time = 0.0
        target_angle = np.pi  # Target upside-down position
        
        action_list = actions.tolist()
        while self.digital_twin.steps < self.simulation_steps:
            if self.digital_twin.steps%self.step_resolution == 0 and len(action_list) > 0:
                action = action_list.pop(0)
                direction, duration = self.digital_twin.action_map[action]
                self.digital_twin.perform_action(direction, duration)
            
            theta, theta_dot, x_pivot,_ = self.digital_twin.step()
            total_time += self.simulation_delta_t
            
            # Calculate deviation from target upside-down position
            angle_deviation = abs(theta - target_angle)
            
            # Track maximum deviation from target
            if angle_deviation > max_deviation:
                max_deviation = angle_deviation
            
            # Reward stability near upside-down position
            if angle_deviation < 0.2:  # Consider stable if within 0.2 radians of π
                stable_time += self.simulation_delta_t
                stability_score += (0.2 - angle_deviation) * self.simulation_delta_t
            
            # Penalize if pivot goes out of bounds
            if abs(self.digital_twin.x_pivot) > 99:
                return -100
        
        # Calculate final fitness score
        # Components:
        # 1. Stability score near upside-down position (higher is better)
        # 2. Penalty for maximum deviation from target (lower is better)
        # 3. Ratio of stable time to total time (higher is better)
        # 4. Penalty if not exactly 10 movements
        movement_penalty = abs(len(actions) - 10) * 10  # Penalize if not exactly 10 movements
        stability_ratio = stable_time / total_time if total_time > 0 else 0
        
        final_score = (stability_score * 100) - (max_deviation * 50) + (stability_ratio * 200) - movement_penalty
        
        return final_score

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
        # Ensure no zeros in offspring
        offspring = np.where(offspring == 0, 1, offspring)
        return offspring

    def mutate(self, individual, mutation_rate=0.2):
        """Mutate an individual's actions with a given mutation rate."""
        for i in range(self.num_steps):
            if random.random() < mutation_rate:
                # Ensure mutated value is never zero
                individual[i] = random.randint(1, self.num_actions - 1)
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
ga = InvertedPendulumGA(population_size=60, num_actions=25, simulation_duration=0.25, action_resolution=0.025, simulation_delta_t=0.025)
best_solution = ga.optimize(num_generations=1000, fitness_threshold=np.pi)

print("Best Solution:", best_solution)

