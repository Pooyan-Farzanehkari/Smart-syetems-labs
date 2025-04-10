import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Path to the CSV file
csv_file_path = 'test_data.csv'

# Read the data from the CSV file
df = pd.read_csv(csv_file_path)

# Convert 'time' from milliseconds to seconds for plotting
df['time_seconds'] = df['time'] / 1000

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(df['time_seconds'], df['theta'])
plt.xlabel('Time (s)')
plt.ylabel('Theta')
plt.title('Theta over Time')
plt.grid(True)

# Set the y-axis labels to display every second
plt.yticks(range(0, int(df['time_seconds'].max()) + 1, 1))

# Show the plot
plt.show()
# Compute the FFT of the signal
theta = df['theta'].values
n = len(theta)
dt = (df['time_seconds'].iloc[1] - df['time_seconds'].iloc[0])  # Time step
freq = np.fft.fftfreq(n, d=dt)
fft_values = np.fft.fft(theta)

# Plot the FFT (magnitude spectrum)
plt.figure(figsize=(10, 5))
plt.plot(freq[:n // 2], np.abs(fft_values[:n // 2]))  # Only plot the positive frequencies
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('FFT of Theta Signal')
plt.grid(True)
plt.show()