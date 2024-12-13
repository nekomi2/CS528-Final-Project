import os
import sys
import time
import csv
import threading
import serial
import joblib
from queue import Queue
from serial_reader import SerialReaderThread
from data_handler import DataHandlerThread
from drone_controller import DroneController
import logging
from logging.handlers import RotatingFileHandler
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = RotatingFileHandler("drone_control.log", maxBytes=5 * 1024 * 1024)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
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
        data_queue,
        window_size=800,
        confidence_threshold=0.7,
        cooldown=0,
    ):
        super().__init__()
        self.data_buffer = data_buffer
        self.data_lock = data_lock
        self.model_path = model_path
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.cooldown = cooldown
        self.data_queue = data_queue

        try:
            self.svm_classifier = joblib.load(model_path)
            if not hasattr(self.svm_classifier, "predict_proba"):
                logging.error("Model does not support probability estimates. Exiting.")
                sys.exit(1)
            logging.info(f"Loaded SVM model from '{model_path}'.")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            sys.exit(1)

        self.label_lookup = {
            0: "Up",
            1: "Down",
            2: "Left",
            3: "Right",
            4: "Neutral",
            5: "Clap",
        }

        self.drone = drone_controller

        self.stop_event = threading.Event()

    def run(self):
        logging.info("PredictionThread started.")
        while not self.stop_event.is_set():
            try:
                if self.data_queue.empty():
                    time.sleep(0.1)
                    continue
                with self.data_lock:
                    if len(self.data_buffer) < self.window_size:
                        continue
                    window_data = np.array(self.data_buffer[-self.window_size :])
                min_vals = window_data.min(axis=0)
                max_vals = window_data.max(axis=0)
                range_vals = max_vals - min_vals
                range_vals[range_vals == 0] = 1e-6  # Prevent division by zero
                window_data_norm = (window_data - min_vals) / range_vals

                window_data_flat = window_data_norm.flatten().reshape(1, -1)

                # Make predictions
                try:
                    y_proba = self.svm_classifier.predict_proba(window_data_flat)
                    y_pred = self.svm_classifier.predict(window_data_flat)
                except Exception as e:
                    logging.error(f"Prediction error: {e}")
                    continue

                confidence_score = np.max(y_proba)
                predicted_label = int(y_pred[0])
                predicted_gesture = self.label_lookup.get(predicted_label, "Unknown")

                logging.info(
                    f"Predicted Gesture: {predicted_gesture}, Confidence: {confidence_score:.2f}"
                )

                # Check confidence and gesture type
                if (
                    predicted_gesture not in ["Neutral", "Unknown"]
                    and confidence_score >= self.confidence_threshold
                ):
                    self.execute_command(predicted_gesture)

                    logging.info(
                        f"Non-neutral gesture detected. Waiting for {self.cooldown} seconds..."
                    )
                    time.sleep(self.cooldown)

                    # Clear data to avoid sticky detections
                    with self.data_lock:
                        self.data_buffer.clear()
                    with self.data_queue.mutex:
                        self.data_queue.queue.clear()

                    logging.info("Resuming predictions...")

            except Exception as e:
                logging.error(f"Unexpected error in PredictionThread: {e}")

    def execute_command(self, gesture):
        """
        Execute drone command based on the predicted gesture.
        """
        logging.info(f"Executing command for gesture: {gesture}")

        try:
            if gesture == "Up":
                self.drone.move_up(20)
            elif gesture == "Down":
                self.drone.move_down(20)
            elif gesture == "Left":
                self.drone.rotate_left()
            elif gesture == "Right":
                self.drone.rotate_right()
            else:
                logging.warning(f"Unknown gesture: {gesture}")
        except Exception as e:
            logging.error(f"Error executing drone command '{gesture}': {e}")

    def stop(self):
        self.stop_event.set()


def main():
    try:
        csv_file_name = "data.csv"
        csv_file = open(csv_file_name, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
        )
        logging.info(f"Data will be saved in file: {csv_file_name}")
    except Exception as e:
        logging.error(f"Failed to open file '{csv_file_name}' for writing: {e}")
        sys.exit(1)

    model_path = "svm_model.joblib"
    if not os.path.exists(model_path):
        logging.error(f"Model file '{model_path}' not found.")
        csv_file.close()
        sys.exit(1)

    try:
        drone = DroneController(max_range=60)
        drone.drone.take_off()
        logging.info("Drone takeoff initiated.")
        time.sleep(5)
        drone.set_position(z=20)
        logging.info(f"Drone takeoff completed. Current position: {drone.position}")
    except Exception as e:
        logging.error(f"Failed to initialize drone: {e}")
        csv_file.close()
        sys.exit(1)

    try:
        ser = serial.Serial("/dev/tty.usbserial-1130", 115200, timeout=1)
        ser.flushInput()
        logging.info("Serial port '/dev/tty.usbserial-1130' opened successfully.")
    except Exception as e:
        logging.error(f"Could not open serial port: {e}")
        drone.land_and_disconnect()
        csv_file.close()
        sys.exit(1)

    serial_reader_thread = SerialReaderThread(ser, data_queue)
    serial_reader_thread.start()
    logging.info("SerialReaderThread started.")

    data_handler_thread = DataHandlerThread(
        data_queue, data_buffer, data_lock, csv_writer, max_buffer_size=2000
    )
    data_handler_thread.start()
    logging.info("DataHandlerThread started.")

    prediction_thread = PredictionThread(
        data_buffer,
        data_lock,
        model_path,
        drone,
        data_queue,
    )
    prediction_thread.start()
    logging.info("PredictionThread started.")

    # Startup takes a long time, needs to send a buffer command to prevent disconnecting after 15 seconds
    drone.move_up(20)

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
