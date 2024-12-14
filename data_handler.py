import threading
from queue import Empty


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
                data = self.data_queue.get(timeout=0.5)
                self.csv_writer.writerow(data)
                with self.data_lock:
                    self.data_buffer.append(data)
                    if len(self.data_buffer) > self.max_buffer_size:
                        self.data_buffer.pop(0)
            except Empty:
                continue
            except Exception as e:
                print(f"Error in DataHandlerThread: {e}")

    def stop(self):
        self.stop_event.set()
