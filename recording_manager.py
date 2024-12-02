import os
import csv
import time
import threading


class RecordingManager(threading.Thread):
    def __init__(self, folder):
        super().__init__()
        self.folder = folder
        self.is_recording = False
        self.countdown = 0
        self.file_number = 0
        self.current_file = None
        self.csv_writer = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            with self.lock:
                filename = os.path.join(
                    self.folder, f"{self.folder}_{self.file_number:02}.csv"
                )
                try:
                    self.current_file = open(filename, "w", newline="")
                    self.csv_writer = csv.writer(self.current_file)
                    self.csv_writer.writerow(
                        ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
                    )
                    self.is_recording = True
                    self.countdown = 4
                    print(f"Started recording: {filename}")
                except Exception as e:
                    print(f"Failed to open file {filename} for writing: {e}")
                    self.is_recording = False

            for i in range(4, 0, -1):
                with self.lock:
                    self.countdown = i
                time.sleep(1)
                if self.stop_event.is_set():
                    break

            with self.lock:
                self.is_recording = False
                print(f"Stopped recording: {filename}")

            with self.lock:
                self.countdown = 2

            for i in range(2, 0, -1):
                with self.lock:
                    self.countdown = i
                time.sleep(1)
                if self.stop_event.is_set():
                    break

            with self.lock:
                if self.current_file:
                    self.current_file.close()
                    self.current_file = None
                    self.csv_writer = None
                self.file_number += 1

    def write_data(self, data):
        with self.lock:
            if self.is_recording and self.csv_writer:
                try:
                    self.csv_writer.writerow(data)
                    self.current_file.flush()
                except Exception as e:
                    print(f"Error writing data to CSV: {e}")

    def get_status(self):
        with self.lock:
            return self.is_recording, self.countdown

    def stop(self):
        self.stop_event.set()
        self.join()
        with self.lock:
            if self.current_file:
                self.current_file.close()
                self.current_file = None
                self.csv_writer = None
