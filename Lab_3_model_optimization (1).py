import numpy as np
from Digital_twin import DigitalTwin
import pandas as pd

digital_twin = DigitalTwin()
# digital_twin.get_theta_double_dot()
# Path to the CSV file
csv_file_path = '/Users/pikpik/Desktop/sss/Smart-syetems-labs/theta.csv'
df = pd.read_csv(csv_file_path)
df_time = df['pcTime'][20:] 
df_theta = df['theta'][20:]

#Process dt_theta sush that it is translated to radians.

def find_initial_state(df_theta, df_time):
     # Find the initial condions of theta and theta_dot in the data
     theta = 92.0 * np.pi / 180  # Convert degrees to radians
     theta_dot = 0
     return theta, theta_dot

#Initial conditions based on the recorded data
theta, theta_dot = find_initial_state(df_theta, df_time)
#should be the same as your recording, sim_time is equal to total time of the recording
delta_t = 0.025  # Time step
sim_time = 0.025 * len(df_theta)  # Total time of the simulation

# Define improved parameter ranges with higher resolution
I_range = np.linspace(0.001, 0.001, 10)      # Reduced upper bound, increased resolution
c_c_range = np.linspace(0.0001, 0.005, 10)    # Increased upper bound, increased resolution  
g = 9.81                                   # Gravity constant
mp_range = np.linspace(0.01, 0.4, 20)       # Expanded range in both directions, increased resolution
l = 0.35

def simulate_potential_model(theta, theta_dot, I, c_c, g, mp, theta_measurements):
    digital_twin.I = I
    digital_twin.c_c = c_c
    digital_twin.g = g
    digital_twin.mp = mp
    digital_twin.l = l

    sim_measurements = []
    for _ in range(len(theta_measurements)):
        theta_double_dot = digital_twin.get_theta_double_dot(theta, theta_dot)
        theta_dot += theta_double_dot * delta_t
        theta += theta_dot * delta_t
        sim_measurements.append(theta)

    sim_measurements = np.array(sim_measurements)
    theta_measurements = np.array(theta_measurements)

    # 1. Basic RMSE
    rmse = np.sqrt(np.mean((theta_measurements - sim_measurements) ** 2))

    # 2. Frequency Domain Analysis
    # Calculate frequencies
    freq = np.fft.fftfreq(len(theta_measurements), d=delta_t)
    fft_measured = np.fft.fft(theta_measurements)
    fft_sim = np.fft.fft(sim_measurements)
    
    # Only consider positive frequencies up to Nyquist frequency
    pos_freq_mask = freq > 0
    pos_freq = freq[pos_freq_mask]
    fft_measured_pos = fft_measured[pos_freq_mask]
    fft_sim_pos = fft_sim[pos_freq_mask]

    # Find dominant frequencies (top 3 peaks)
    N = 3  # Number of dominant frequencies to consider
    measured_peaks = np.argsort(np.abs(fft_measured_pos))[-N:]
    sim_peaks = np.argsort(np.abs(fft_sim_pos))[-N:]
    
    # Compare magnitude and phase of dominant frequencies
    freq_error = np.mean(np.abs(measured_peaks - sim_peaks))
    magnitude_error = np.mean(np.abs(np.abs(fft_measured_pos[measured_peaks]) - 
                                   np.abs(fft_sim_pos[sim_peaks])))
    phase_error_freq = np.mean(np.abs(np.angle(fft_measured_pos[measured_peaks]) - 
                                    np.angle(fft_sim_pos[sim_peaks])))

    # 3. Natural Frequency Analysis
    natural_freq = np.sqrt(g/l)  # theoretical natural frequency
    measured_dom_freq = pos_freq[np.argmax(np.abs(fft_measured_pos))]
    sim_dom_freq = pos_freq[np.argmax(np.abs(fft_sim_pos))]
    
    freq_penalty = np.abs(measured_dom_freq - sim_dom_freq)
    natural_freq_penalty = np.abs(measured_dom_freq - natural_freq)

    # 4. Time-windowed Analysis
    window_size = len(theta_measurements) // 4
    n_windows = 4
    window_errors = []
    
    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size
        
        window_measured = theta_measurements[start_idx:end_idx]
        window_sim = sim_measurements[start_idx:end_idx]
        
        # Calculate FFT for window
        fft_measured_window = np.fft.fft(window_measured)
        fft_sim_window = np.fft.fft(window_sim)
        
        # Calculate error for this window
        window_error = np.mean(np.abs(fft_measured_window - fft_sim_window))
        window_errors.append(window_error)
    
    # Time-varying frequency error
    freq_evolution_error = np.mean(window_errors)

    # 5. Phase Alignment (using cross-correlation)
    cross_corr = np.correlate(theta_measurements - np.mean(theta_measurements),
                            sim_measurements - np.mean(sim_measurements), mode='full')
    max_corr = np.max(cross_corr)
    phase_error_time = 1 - (max_corr / (np.std(theta_measurements) * 
                                       np.std(sim_measurements) * 
                                       len(theta_measurements)))

    # Combine all error components with weights
    total_error = (
        0.15 * rmse +                    # Basic time-domain error
        0.20 * freq_error +              # Dominant frequency matching
        0.15 * magnitude_error +         # Frequency magnitude matching
        0.10 * phase_error_freq +        # Frequency domain phase matching
        0.10 * freq_penalty +            # Dominant frequency difference
        0.10 * natural_freq_penalty +    # Natural frequency matching
        0.10 * freq_evolution_error +    # Time-windowed frequency analysis
        0.10 * phase_error_time          # Time-domain phase matching
    )

    return total_error


#example usage
exp_err = simulate_potential_model(np.pi-0.5, -0.1, 0.16, 1.1, 9.81 ,0.3, df_theta)
# Initialize variables to store the best parameters and the lowest error found
best_params = None
lowest_error = float('inf')

# Nested loops to go through each combination of parameters
for I in I_range:
        for c_c in c_c_range:
            for mp in mp_range:
                error = simulate_potential_model(theta, theta_dot, I, c_c, g, mp, df_theta)
                if error < lowest_error:
                    lowest_error = error
                    print(error, "found a better error")
                    best_params = (I, c_c, mp)

print("Best Parameters:", best_params)
print("Lowest Error:", lowest_error)
print("expected Error: ", exp_err)

import matplotlib.pyplot as plt

# Simulate theta measurements using the best parameters
best_c_air, best_c_c, best_l = best_params
simulated_theta = []
theta_sim = theta
theta_dot_sim = theta_dot
simulated_theta_dot = []
for i in range(len(df_theta)):
     theta_double_dot_sim = digital_twin.get_theta_double_dot(theta_sim, theta_dot_sim)
     theta_dot_sim += theta_double_dot_sim * delta_t
     theta_sim += theta_dot_sim * delta_t

     simulated_theta.append(theta_sim)
     simulated_theta_dot.append(theta_dot_sim)

# Plot the real and simulated theta
plt.figure(figsize=(10, 6))
plt.plot(df_time, df_theta, label='Real Theta', color='blue')
plt.plot(df_time, simulated_theta, label='Simulated Theta', color='red', linestyle='--')
plt.plot(df_time, simulated_theta_dot, label='Simulated Theta Dot', color='green', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Theta (radians)')
plt.title('Comparison of Real and Simulated Theta')
plt.legend()
plt.grid()
plt.show()



import matplotlib.animation as animation

# Create a figure for the animation
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect('equal')
ax.grid()

# Initialize the pendulum line
line, = ax.plot([], [], 'o-', lw=2, color='blue')

# Function to initialize the animation
def init():
     line.set_data([], [])
     return line,

# Function to update the animation frame
def update(frame):
     x = best_l * np.sin(simulated_theta[frame])
     y = -best_l * np.cos(simulated_theta[frame])
     line.set_data([0, x], [0, y])
     return line,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(simulated_theta), init_func=init, blit=True, interval=delta_t * 1000)

# Show the animation
plt.show()