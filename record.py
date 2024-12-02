import os
import sys
import queue
import serial
from recording_manager import RecordingManager
from serial_reader import SerialReaderThread
from plot_live import plot_live
from utils import graceful_exit
import signal


def main():
    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)

    folder_name = input("Enter the name for the data folder: ").strip()
    if not folder_name:
        print("Folder name cannot be empty.")
        sys.exit(1)

    try:
        os.makedirs(folder_name, exist_ok=True)
        print(f"Data will be saved in folder: {folder_name}")
    except Exception as e:
        print(f"Failed to create folder '{folder_name}': {e}")
        sys.exit(1)

    data_queue = queue.Queue()

    recording_manager = RecordingManager(folder=folder_name)
    recording_manager.start()

    try:
        ser = serial.Serial("/dev/tty.usbserial-1130", 115200)
    except Exception as e:
        print(f"Could not open serial port: {e}")
        recording_manager.stop()
        sys.exit(1)

    serial_thread = SerialReaderThread(ser, data_queue)
    serial_thread.start()

    try:
        plot_live(recording_manager, data_queue)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        serial_thread.stop()
        serial_thread.join()
        ser.close()
        recording_manager.stop()
        print("Recording stopped. Exiting.")


if __name__ == "__main__":
    main()
