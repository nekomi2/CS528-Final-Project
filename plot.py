import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_file", type=str, help="Path to the CSV file containing sensor data."
    )
    parser.add_argument(
        "--sampling_rate",
        type=float,
        default=100.0,
    )
    return parser.parse_args()


def load_data(csv_file):
    if not os.path.isfile(csv_file):
        print(f"Error: File '{csv_file}' does not exist.")
        sys.exit(1)

    try:
        data = pd.read_csv(csv_file)
        required_columns = [
            "accel_x",
            "accel_y",
            "accel_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
        ]
        if not all(col in data.columns for col in required_columns):
            print(
                f"Error: CSV file must contain the following columns: {', '.join(required_columns)}"
            )
            sys.exit(1)
        return data
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)


def perform_fft(data, sampling_rate):
    fft_results = {}
    N = len(data)
    T = 1.0 / sampling_rate
    freq = fftfreq(N, T)[: N // 2]

    for col in data.columns:
        yf = fft(data[col])
        yf_magnitude = 2.0 / N * np.abs(yf[0 : N // 2])
        fft_results[col] = yf_magnitude
    return freq, fft_results


def plot_combined(time, accel_data, gyro_data, freq, fft_results):
    fig, axs = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle("MPU6050 Sensor Data Analysis", fontsize=18)

    accel_labels = ["Accel X", "Accel Y", "Accel Z"]
    gyro_labels = ["Gyro X", "Gyro Y", "Gyro Z"]
    accel_colors = ["r", "g", "b"]
    gyro_colors = ["c", "m", "y"]

    ax1 = axs[0, 0]
    for i, col in enumerate(accel_data.columns):
        ax1.plot(time, accel_data[col], color=accel_colors[i], label=accel_labels[i])
    ax1.set_title("Accelerometer Time-Domain Signals", fontsize=14)
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_ylabel("Acceleration (g)", fontsize=12)
    ax1.legend()
    ax1.grid(True)

    
    ax2 = axs[0, 1]
    for i, col in enumerate(gyro_data.columns):
        ax2.plot(time, gyro_data[col], color=gyro_colors[i], label=gyro_labels[i])
    ax2.set_title("Gyroscope Time-Domain Signals", fontsize=14)
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Angular Velocity (deg/s)", fontsize=12)
    ax2.legend()
    ax2.grid(True)

    
    ax3 = axs[1, 0]
    for i, col in enumerate(accel_data.columns):
        ax3.plot(freq, fft_results[col], color=accel_colors[i], label=accel_labels[i])
    ax3.set_title("Accelerometer Frequency Spectrum (FFT)", fontsize=14)
    ax3.set_xlabel("Frequency (Hz)", fontsize=12)
    ax3.set_ylabel("Magnitude", fontsize=12)
    ax3.legend()
    ax3.grid(True)

    
    ax4 = axs[1, 1]
    for i, col in enumerate(gyro_data.columns):
        ax4.plot(freq, fft_results[col], color=gyro_colors[i], label=gyro_labels[i])
    ax4.set_title("Gyroscope Frequency Spectrum (FFT)", fontsize=14)
    ax4.set_xlabel("Frequency (Hz)", fontsize=12)
    ax4.set_ylabel("Magnitude", fontsize=12)
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout(
        rect=[0, 0.03, 1, 0.95]
    )  
    plt.show()


def main():
    args = parse_arguments()
    csv_file = args.csv_file
    sampling_rate = args.sampling_rate
    data = load_data(csv_file)
    num_samples = len(data)
    time = np.arange(num_samples) / sampling_rate
    accel_columns = ["accel_x", "accel_y", "accel_z"]
    gyro_columns = ["gyro_x", "gyro_y", "gyro_z"]
    accel_data = data[accel_columns]
    gyro_data = data[gyro_columns]
    freq, fft_results = perform_fft(data, sampling_rate)
    plot_combined(time, accel_data, gyro_data, freq, fft_results)


if __name__ == "__main__":
    main()
