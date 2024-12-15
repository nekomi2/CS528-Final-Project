import os
import sys
import time
import csv
import threading
import serial
from queue import Queue
from serial_reader import SerialReaderThread
from data_handler import DataHandlerThread
from drone_controller import DroneController
import logging
from logging.handlers import RotatingFileHandler
from prediction import PredictionThread

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
        # stunlock
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
