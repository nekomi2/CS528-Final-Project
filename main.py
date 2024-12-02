# drone_control.py

import os
import sys
import time
import csv
import threading
import signal
import serial
import joblib
from queue import Queue, Empty
from serial_reader import SerialReaderThread
from data_handler import DataHandlerThread
from drone_controller import DroneController
import logging
from logging.handlers import RotatingFileHandler
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(
    "drone_control.log", maxBytes=5 * 1024 * 1024, backupCount=3
)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)

# Also add a stream handler for console output
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Shared data structures
data_buffer = []
data_lock = threading.Lock()
data_queue = Queue()


class PredictionThread(threading.Thread):
    def __init__(
        self,
        data_buffer,
        data_lock,
        model_path,
        drone_controller,
        window_size=900,
        step_size=100,
        confidence_threshold=0.5,
        cooldown=1,
    ):
        super().__init__()
        self.data_buffer = data_buffer
        self.data_lock = data_lock
        self.model_path = model_path
        self.window_size = window_size
        self.step_size = step_size
        self.stop_event = threading.Event()
        self.confidence_threshold = confidence_threshold
        self.cooldown = cooldown

        try:
            self.svm_classifier = joblib.load(model_path)
            if not hasattr(self.svm_classifier, "predict_proba"):
                logging.error("Model does not support probability estimates. Exiting.")
                sys.exit(1)
            logging.info(f"Loaded SVM model from '{model_path}'.")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            sys.exit(1)

        self.label_lookup = {0: "Up", 1: "Down", 2: "Left", 3: "Right", 4: "Neutral"}

        self.last_index = 0
        self.last_command_time = 0

        self.drone = drone_controller

    def run(self):
        logging.info("PredictionThread started.")
        while not self.stop_event.is_set():
            try:
                time.sleep(0.1)

                with self.data_lock:
                    data_length = len(self.data_buffer)

                if data_length - self.last_index >= self.window_size:
                    with self.data_lock:
                        data_array = np.array(self.data_buffer)

                    for start in range(
                        self.last_index,
                        data_length - self.window_size + 1,
                        self.step_size,
                    ):
                        end = start + self.window_size
                        window_data = data_array[start:end]

                        min_vals = window_data.min(axis=0)
                        max_vals = window_data.max(axis=0)
                        range_vals = max_vals - min_vals
                        range_vals[range_vals == 0] = 1e-6
                        window_data_norm = (window_data - min_vals) / range_vals

                        window_data_flat = window_data_norm.flatten().reshape(1, -1)

                        try:
                            y_proba = self.svm_classifier.predict_proba(
                                window_data_flat
                            )
                            y_pred = self.svm_classifier.predict(window_data_flat)
                        except Exception as e:
                            logging.error(f"Prediction error: {e}")
                            continue

                        confidence_score = np.max(y_proba)
                        predicted_gesture = self.label_lookup.get(
                            int(y_pred[0]), "Unknown"
                        )

                        logging.info(
                            f"Predicted Gesture: {predicted_gesture}, Confidence: {confidence_score:.2f}"
                        )

                        current_time = time.time()
                        logging.info(
                            f"{predicted_gesture} and {confidence_score >= self.confidence_threshold}"
                        )
                        if (
                            predicted_gesture != "Neutral"
                            and confidence_score >= self.confidence_threshold
                            and (current_time - self.last_command_time) >= self.cooldown
                        ):
                            self.execute_command(predicted_gesture)
                            self.last_command_time = current_time

                    self.last_index = data_length - self.window_size + 1

            except Exception as e:
                logging.error(f"Unexpected error in PredictionThread: {e}")

    def execute_command(self, gesture):
        """
        Execute drone command based on the predicted gesture.
        """
        logging.info(f"Executing command for gesture: {gesture}")

        if gesture == "Up":
            # self.drone.move_up(20)
            return
        elif gesture == "Down":
            # self.drone.move_down(20)
            return
        elif gesture == "Left":
            self.drone.rotate_left()
        elif gesture == "Right":
            self.drone.rotate_right()
        else:
            logging.warning(f"Unknown gesture: {gesture}")

    def stop(self):
        self.stop_event.set()


def graceful_exit(signum, frame):
    logging.info("\nShutting down gracefully...")
    raise KeyboardInterrupt


def main():
    # Set up signal handlers
    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)

    # Define CSV file name
    csv_file_name = "data.csv"

    # Open CSV file for writing
    try:
        csv_file = open(csv_file_name, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
        )
        logging.info(f"Data will be saved in file: {csv_file_name}")
    except Exception as e:
        logging.error(f"Failed to open file '{csv_file_name}' for writing: {e}")
        sys.exit(1)

    # Load the SVM model
    model_path = "svm_model.joblib"
    if not os.path.exists(model_path):
        logging.error(f"Model file '{model_path}' not found.")
        csv_file.close()
        sys.exit(1)

    # Initialize the drone controller
    try:
        drone = DroneController(max_range=60)
        drone.drone.take_off()
        logging.info("Drone takeoff initiated.")
        time.sleep(5)  # Wait for takeoff to stabilize
        drone.set_position(z=10)  # Assume takeoff to 10 cm
        logging.info(f"Drone takeoff completed. Current position: {drone.position}")
    except Exception as e:
        logging.error(f"Failed to initialize drone: {e}")
        csv_file.close()
        sys.exit(1)

    # Set up serial port
    try:
        ser = serial.Serial("/dev/tty.usbserial-1130", 115200, timeout=1)
        ser.flushInput()
        logging.info(f"Serial port '/dev/tty.usbserial-1130' opened successfully.")
    except Exception as e:
        logging.error(f"Could not open serial port: {e}")
        drone.land_and_disconnect()
        csv_file.close()
        sys.exit(1)

    # Start SerialReaderThread
    serial_reader_thread = SerialReaderThread(ser, data_queue)
    serial_reader_thread.start()
    logging.info("SerialReaderThread started.")

    # Start DataHandlerThread
    data_handler_thread = DataHandlerThread(
        data_queue, data_buffer, data_lock, csv_writer, max_buffer_size=2000
    )
    data_handler_thread.start()
    logging.info("DataHandlerThread started.")

    # Start PredictionThread
    prediction_thread = PredictionThread(
        data_buffer,
        data_lock,
        model_path,
        drone,
    )
    prediction_thread.start()
    logging.info("PredictionThread started.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("\nInterrupted by user.")
    finally:
        serial_reader_thread.stop()
        serial_reader_thread.join()
        logging.info("SerialReaderThread stopped.")

        data_handler_thread.stop()
        data_handler_thread.join()
        logging.info("DataHandlerThread stopped.")

        prediction_thread.stop()
        prediction_thread.join()
        logging.info("PredictionThread stopped.")

        ser.close()
        logging.info("Serial port closed.")
        csv_file.close()
        logging.info("CSV file closed.")

        try:
            drone.land_and_disconnect()
        except Exception as e:
            logging.error(f"Error during drone shutdown: {e}")

        logging.info("Recording stopped. Drone landed and disconnected. Exiting.")


if __name__ == "__main__":
    main()
