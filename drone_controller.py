import threading
import logging
from TelloDroneBuilder import (
    TelloDroneBuilder,
)
import time


class DroneController:
    def __init__(self, max_range=40):
        """
        Initialize the DroneController with a maximum range.

        Args:
            max_range (int): Maximum movement range in centimeters.
        """
        self.drone = TelloDroneBuilder()
        self.lock = threading.Lock()
        self.max_range = max_range
        self.position = {"x": 0, "y": 0, "z": 0}
        logging.info(f"DroneController initialized with max_range={self.max_range} cm.")

    def set_position(self, x=0, y=0, z=0):
        with self.lock:
            self.position = {"x": x, "y": y, "z": z}
            logging.info(f"Drone position set to: {self.position}")

    def move_up(self, distance):
        with self.lock:
            new_z = self.position["z"] + distance
            logging.debug(
                f"Attempting to move up by {distance} cm. Current Z: {self.position['z']}, New Z: {new_z}"
            )
            if new_z > self.max_range:
                distance = self.max_range - self.position["z"]
                if distance <= 0:
                    logging.warning("Max upward range reached. Cannot move up further.")
                    return
                logging.info(
                    f"Adjusting move_up distance to {distance} cm to stay within range."
                )
            self.drone.move_up(distance)
            self.position["z"] += distance
            logging.info(
                f"Moved up by {distance} cm. Current position: {self.position}"
            )

    def move_down(self, distance):
        with self.lock:
            new_z = self.position["z"] - distance
            logging.debug(
                f"Attempting to move down by {distance} cm. Current Z: {self.position['z']}, New Z: {new_z}"
            )
            if new_z < -self.max_range:
                distance = self.position["z"] + self.max_range
                if distance <= 0:
                    logging.warning(
                        "Max downward range reached. Cannot move down further."
                    )
                    return
                logging.info(
                    f"Adjusting move_down distance to {distance} cm to stay within range."
                )
            self.drone.move_down(distance)
            self.position["z"] -= distance
            logging.info(
                f"Moved down by {distance} cm. Current position: {self.position}"
            )

    def move_left(self, distance):
        with self.lock:
            new_y = self.position["y"] - distance
            logging.debug(
                f"Attempting to move left by {distance} cm. Current Y: {self.position['y']}, New Y: {new_y}"
            )
            if abs(new_y) > self.max_range:
                distance = (
                    self.position["y"] + self.max_range
                    if new_y < -self.max_range
                    else distance
                )
                if distance <= 0:
                    logging.warning("Max left range reached. Cannot move left further.")
                    return
                logging.info(
                    f"Adjusting move_left distance to {distance} cm to stay within range."
                )
            self.drone.move_left(distance)
            self.position["y"] -= distance
            logging.info(
                f"Moved left by {distance} cm. Current position: {self.position}"
            )

    def move_right(self, distance):
        with self.lock:
            new_y = self.position["y"] + distance
            logging.debug(
                f"Attempting to move right by {distance} cm. Current Y: {self.position['y']}, New Y: {new_y}"
            )
            if abs(new_y) > self.max_range:
                distance = (
                    self.max_range - self.position["y"]
                    if new_y > self.max_range
                    else distance
                )
                if distance <= 0:
                    logging.warning(
                        "Max right range reached. Cannot move right further."
                    )
                    return
                logging.info(
                    f"Adjusting move_right distance to {distance} cm to stay within range."
                )
            self.drone.move_right(distance)
            self.position["y"] += distance
            logging.info(
                f"Moved right by {distance} cm. Current position: {self.position}"
            )

    def rotate_left(self):
        with self.lock:
            self.drone.rotate(-90)
            logging.info("Rotated left by 90 degrees.")

    def rotate_right(self):
        with self.lock:
            self.drone.rotate(90)
            logging.info("Rotated right by 90 degrees.")

    def flip(self):
        with self.lock:
            self.drone.flip_back()
            logging.info("Flipping ")

    def fetch(self):
        with self.lock:
            self.drone.fly_square(20)
            logging.info("Spinning")

    def land_and_disconnect(self, max_retries=3):
        with self.lock:
            logging.info("Attempting to land the drone.")
            for attempt in range(1, max_retries + 1):
                logging.info(f"Send command: 'land' (Attempt {attempt})")
                self.drone.land()
                logging.info(f"Sent 'land' command (Attempt {attempt}).")
                time.sleep(2)
            logging.critical(
                "Failed to land the drone after multiple attempts. Initiating emergency stop."
            )
            self.emergency_stop()

    def emergency_stop(self):
        with self.lock:
            logging.info("Emergency stop initiated.")
            self.drone.emergency()
            logging.info("Sent 'emergency' command.")
            self.drone.disconnect()
            logging.info("Drone performed emergency stop and disconnected.")
