import logging
import sys
import threading
import time
import joblib
import numpy as np


class PredictionThread(threading.Thread):
    def __init__(
        self,
        data_buffer,
        data_lock,
        model_path,
        drone_controller,
        data_queue,
        window_size=800,
        confidence_threshold=0.4,
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
            6: "Fetch",
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
            elif gesture == "Clap":
                self.drone.flip()
            elif gesture == "Fetch":
                self.drone.fetch()
            else:
                logging.warning(f"Unknown gesture: {gesture}")
        except Exception as e:
            logging.error(f"Error executing drone command '{gesture}': {e}")

    def stop(self):
        self.stop_event.set()
