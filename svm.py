import os
import sys
import time
import csv
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import signal
import serial
import joblib

# Shared data structures
data_buffer = []
data_lock = threading.Lock()


class SerialReaderThread(threading.Thread):
    def __init__(self, ser, data_buffer, data_lock, csv_writer):
        super().__init__()
        self.ser = ser
        self.data_buffer = data_buffer
        self.data_lock = data_lock
        self.csv_writer = csv_writer
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            try:
                raw_data = self.ser.readline()
                decoded_data = raw_data.decode("utf-8").strip()

                # Process the data
                if ": " in decoded_data:
                    data_str = decoded_data.split(": ")[1]
                    data_values = data_str.split(",")
                    if len(data_values) == 6:
                        try:
                            data = list(map(float, data_values))
                        except ValueError:
                            print(f"Non-float value encountered: {data_values}")
                            continue

                        # Write to CSV
                        self.csv_writer.writerow(data)

                        # Append data to the shared buffer
                        with self.data_lock:
                            self.data_buffer.append(data)
                        # Debug statement (optional)
                        # print(f"Data appended to buffer: {data}")
                    else:
                        print(f"Unexpected data length: {len(data_values)}")
                else:
                    print(f"Unexpected data format: {decoded_data}")

            except Exception as e:
                print(f"Error reading from serial port: {e}")

    def stop(self):
        self.stop_event.set()


class PredictionThread(threading.Thread):
    def __init__(
        self, data_buffer, data_lock, model_path, window_size=900, step_size=100
    ):
        super().__init__()
        self.data_buffer = data_buffer
        self.data_lock = data_lock
        self.model_path = model_path
        self.window_size = window_size
        self.step_size = step_size
        self.stop_event = threading.Event()

        # Load the SVM classifier with probability estimates
        try:
            self.svm_classifier = joblib.load(model_path)
            if not hasattr(self.svm_classifier, "predict_proba"):
                print("Model does not support probability estimates. Exiting.")
                sys.exit(1)
        except Exception as e:
            print(f"Failed to load model: {e}")
            sys.exit(1)

        # Gesture label mapping
        self.label_lookup = {0: "Up", 1: "Down", 2: "Left", 3: "Right", 4: "Neutral"}

        self.last_index = 0  # To keep track of the last processed index

    def run(self):
        while not self.stop_event.is_set():
            # Sleep for a short duration to reduce CPU usage
            time.sleep(0.1)

            with self.data_lock:
                data_length = len(self.data_buffer)

            # Check if there is enough data for at least one window
            if data_length - self.last_index >= self.window_size:
                with self.data_lock:
                    data_array = np.array(self.data_buffer)

                # Process data from last_index to new data_length
                for start in range(
                    self.last_index,
                    data_length - self.window_size + 1,
                    self.step_size,
                ):
                    end = start + self.window_size
                    window_data = data_array[start:end]

                    # Normalize window_data as in training
                    min_vals = window_data.min(axis=0)
                    max_vals = window_data.max(axis=0)
                    range_vals = max_vals - min_vals
                    range_vals[range_vals == 0] = 1e-6  # Prevent division by zero
                    window_data_norm = (window_data - min_vals) / range_vals

                    # Flatten and reshape for prediction
                    window_data_flat = window_data_norm.flatten().reshape(1, -1)

                    # Predict label and get probability
                    try:
                        y_proba = self.svm_classifier.predict_proba(window_data_flat)
                        y_pred = self.svm_classifier.predict(window_data_flat)
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        continue

                    confidence_score = np.max(y_proba)
                    predicted_gesture = self.label_lookup.get(int(y_pred[0]), "Unknown")

                    print(
                        f"Predicted Gesture: {predicted_gesture}, Confidence: {confidence_score:.2f}"
                    )

                # Update the last_index
                self.last_index = data_length - self.window_size + 1

    def stop(self):
        self.stop_event.set()


def plot_live(data_buffer, data_lock):
    max_points = 1000
    # Initialize accel_data and gyro_data with zeros
    accel_data = np.zeros((3, max_points))  # For accel_x, accel_y, accel_z
    gyro_data = np.zeros((3, max_points))  # For gyro_x, gyro_y, gyro_z
    data_index = 0

    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle("MPU6050 Live Data", fontsize=16)

    accel_labels = ["Accel X", "Accel Y", "Accel Z"]
    gyro_labels = ["Gyro X", "Gyro Y", "Gyro Z"]
    y_labels = ["Acceleration (g)", "Angular Velocity (deg/s)"]

    accel_colors = ["r", "g", "b"]
    gyro_colors = ["c", "m", "y"]

    accel_lines = [
        axs[0].plot([], [], color, label=label)[0]
        for color, label in zip(accel_colors, accel_labels)
    ]
    gyro_lines = [
        axs[1].plot([], [], color, label=label)[0]
        for color, label in zip(gyro_colors, gyro_labels)
    ]

    axs[0].set_ylim(-4, 4)
    axs[0].set_xlim(0, max_points - 1)
    axs[0].set_title("Accelerometer Data", fontsize=14)
    axs[0].set_ylabel(y_labels[0], fontsize=12)
    axs[0].legend(loc="upper right")
    axs[0].grid(True)

    axs[1].set_ylim(-500, 500)
    axs[1].set_xlim(0, max_points - 1)
    axs[1].set_title("Gyroscope Data", fontsize=14)
    axs[1].set_ylabel(y_labels[1], fontsize=12)
    axs[1].set_xlabel("Samples", fontsize=12)
    axs[1].legend(loc="upper right")
    axs[1].grid(True)

    def update_plot(frame):
        nonlocal data_index

        with data_lock:
            data_len = len(data_buffer)
            new_data = data_buffer[data_index:data_len]

        if new_data:
            new_data_array = np.array(new_data).T  # Shape: (6, num_new_points)
            num_new_points = new_data_array.shape[1]

            if num_new_points >= max_points:
                # If more new data than max_points, take the last max_points
                accel_data[:, :] = new_data_array[:3, -max_points:]
                gyro_data[:, :] = new_data_array[3:, -max_points:]
            else:
                # Shift data arrays to the left
                shift = num_new_points
                accel_data[:, :-shift] = accel_data[:, shift:]
                accel_data[:, -shift:] = new_data_array[:3]

                gyro_data[:, :-shift] = gyro_data[:, shift:]
                gyro_data[:, -shift:] = new_data_array[3:]

            data_index = data_len

            x_data = np.arange(max_points)

            for i, line in enumerate(accel_lines):
                line.set_data(x_data, accel_data[i])

            for i, line in enumerate(gyro_lines):
                line.set_data(x_data, gyro_data[i])

        return accel_lines + gyro_lines

    ani = FuncAnimation(fig, update_plot, interval=100)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def graceful_exit(signum, frame):
    print("\nShutting down gracefully...")
    raise KeyboardInterrupt


def main():
    # Set up signal handlers
    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)

    # Get CSV file name
    csv_file_name = "data.csv"

    # Open CSV file for writing
    try:
        csv_file = open(csv_file_name, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
        )
        print(f"Data will be saved in file: {csv_file_name}")
    except Exception as e:
        print(f"Failed to open file '{csv_file_name}' for writing: {e}")
        sys.exit(1)

    # Load the SVM model
    model_path = "svm_model.joblib"
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found.")
        csv_file.close()
        sys.exit(1)

    # Set up serial port
    try:
        ser = serial.Serial("/dev/tty.usbserial-1130", 115200, timeout=1)
        # Flush initial data
        ser.flushInput()
    except Exception as e:
        print(f"Could not open serial port: {e}")
        csv_file.close()
        sys.exit(1)

    # Start serial reader thread
    serial_thread = SerialReaderThread(ser, data_buffer, data_lock, csv_writer)
    serial_thread.start()

    # Start prediction thread
    prediction_thread = PredictionThread(data_buffer, data_lock, model_path)
    prediction_thread.start()

    try:
        # Start plotting
        plot_live(data_buffer, data_lock)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # Stop threads and close serial port and file
        serial_thread.stop()
        serial_thread.join()
        prediction_thread.stop()
        prediction_thread.join()
        ser.close()
        csv_file.close()
        print("Recording stopped. Exiting.")


if __name__ == "__main__":
    main()
