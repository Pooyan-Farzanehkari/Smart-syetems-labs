import pygame
import time
import math
from Digital_twin import DigitalTwin
import numpy as np

class PendulumVisualizer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode([1000, 800])
        pygame.display.set_caption("Inverted Pendulum Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (128, 128, 128)
        
    def draw_text(self, text, pos, color=None, font=None):
        if color is None:
            color = self.BLACK
        if font is None:
            font = self.font
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, pos)
        
    def draw_status(self, theta, theta_dot, x_pivot, current_action=None):
        # Draw status panel background
        pygame.draw.rect(self.screen, self.GRAY, (10, 10, 300, 140), border_radius=10)
        pygame.draw.rect(self.screen, self.WHITE, (12, 12, 296, 136), border_radius=9)
        
        # Draw status information
        self.draw_text(f"Angle: {math.degrees(theta):.1f}°", (20, 20))
        self.draw_text(f"Angular Velocity: {theta_dot:.2f}", (20, 50))
        self.draw_text(f"Position: {x_pivot:.1f}", (20, 80))
        
        if current_action:
            direction, duration = current_action
            self.draw_text(f"Action: {direction} ({duration}ms)", (20, 110))
            
    def draw_controls(self):
        # Draw controls panel
        pygame.draw.rect(self.screen, self.GRAY, (690, 10, 300, 140), border_radius=10)
        pygame.draw.rect(self.screen, self.WHITE, (692, 12, 296, 136), border_radius=9)
        
        # Draw control instructions
        controls = [
            "Controls:",
            "R - Restart simulation",
            "SPACE - Pause/Resume",
            "ESC - Quit",
            "→ - Speed up",
            "← - Slow down"
        ]
        
        for i, text in enumerate(controls):
            self.draw_text(text, (700, 20 + i * 22), font=self.small_font)

def run_action_list(action_list):
    # Initialize visualizer and digital twin
    vis = PendulumVisualizer()
    digital_twin = DigitalTwin()
    
    # Simulation parameters
    simulation_delta_t = 0.025
    action_resolution = 0.2
    step_resolution = int(action_resolution / simulation_delta_t)
    simulation_steps = 2.0 / simulation_delta_t  # 2 seconds simulation
    
    # Simulation state
    running = True
    paused = False
    current_action_index = 0
    current_action = None
    speed_multiplier = 1.0
    
    # Print action sequence
    print("\nAction Sequence:")
    for i, action in enumerate(action_list):
        direction, duration = digital_twin.action_map[action]
        print(f"Move {i+1}: {direction} for {duration}ms")
    
    def reset_simulation():
        nonlocal current_action_index, current_action
        digital_twin.theta = 0.
        digital_twin.theta_dot = 0.
        digital_twin.x_pivot = 0.
        digital_twin.steps = 0.
        current_action_index = 0
        current_action = None
        print("\nSimulation reset")
    
    reset_simulation()
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    reset_simulation()
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print("Simulation", "paused" if paused else "resumed")
                elif event.key == pygame.K_RIGHT:
                    speed_multiplier = min(4.0, speed_multiplier * 1.5)
                    print(f"Speed: {speed_multiplier}x")
                elif event.key == pygame.K_LEFT:
                    speed_multiplier = max(0.25, speed_multiplier / 1.5)
                    print(f"Speed: {speed_multiplier}x")
        
        if not paused:
            # Execute actions at the right time
            if digital_twin.steps % step_resolution == 0 and current_action_index < len(action_list):
                action = action_list[current_action_index]
                direction, duration = digital_twin.action_map[action]
                digital_twin.perform_action(direction, duration)
                current_action = (direction, duration)
                print(f"\nExecuting Move {current_action_index + 1}: {direction} for {duration}ms")
                current_action_index += 1
            
            # Step the simulation
            theta, theta_dot, x_pivot, _ = digital_twin.step()
            
            # Check if pendulum has fallen
            if abs(theta) > np.pi/2 or abs(x_pivot) > 99:
                print("\nPendulum has fallen!")
                time.sleep(1)
                reset_simulation()
        
        # Render
        vis.screen.fill(vis.WHITE)
        
        # Draw pendulum
        l = 100
        digital_twin.draw_pendulum(vis.BLACK, l * math.cos(theta), l * math.sin(theta), x_pivot)
        digital_twin.draw_line_and_circles(vis.BLACK, [400, 400], [600, 400])
        
        # Draw UI elements
        vis.draw_status(theta, theta_dot, x_pivot, current_action)
        vis.draw_controls()
        
        # Draw simulation speed
        vis.draw_text(f"Speed: {speed_multiplier:.2f}x", (450, 20))
        if paused:
            vis.draw_text("PAUSED", (450, 50), vis.RED)
        
        pygame.display.flip()
        
        # Control simulation speed
        vis.clock.tick(1/(simulation_delta_t/speed_multiplier))
    
    pygame.quit()
    print("\nSimulation ended")

if __name__ == "__main__":
    try:
        best_solution = [7 ,3 ,1 ,6 ,1 ,6 ,6 ,7 ,5 ,1]
        print("Successfully loaded solution from best_solution.txt")
    except FileNotFoundError:
        print("best_solution.txt not found. Using sample solution...")
        # Sample solution that alternates between left and right movements
        best_solution = [1, 5, 2, 6, 3, 7, 4, 8, 1, 5]
    
    print("\nControls:")
    print("- R: Restart simulation")
    print("- SPACE: Pause/Resume")
    print("- ESC or close window: Quit")
    print("- Right arrow: Speed up")
    print("- Left arrow: Slow down")
    
    # Run the simulation
    run_action_list(best_solution) 