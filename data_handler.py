# data_handler.py

import threading
from queue import Queue, Empty


class DataHandlerThread(threading.Thread):
    def __init__(
        self, data_queue, data_buffer, data_lock, csv_writer, max_buffer_size=2000
    ):
        super().__init__()
        self.data_queue = data_queue
        self.data_buffer = data_buffer
        self.data_lock = data_lock
        self.csv_writer = csv_writer
        self.stop_event = threading.Event()
        self.max_buffer_size = max_buffer_size

    def run(self):
        while not self.stop_event.is_set():
            try:
                # Wait for data with a timeout to allow checking stop_event
                data = self.data_queue.get(timeout=0.5)
                # Write to CSV
                self.csv_writer.writerow(data)

                # Append to data_buffer
                with self.data_lock:
                    self.data_buffer.append(data)
                    # Maintain buffer size
                    if len(self.data_buffer) > self.max_buffer_size:
                        self.data_buffer.pop(0)  # Remove the oldest data point

            except Empty:
                continue  # No data received, continue checking
            except Exception as e:
                print(f"Error in DataHandlerThread: {e}")

    def stop(self):
        self.stop_event.set()
