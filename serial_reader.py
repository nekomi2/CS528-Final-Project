import threading


class SerialReaderThread(threading.Thread):
    def __init__(self, ser, data_queue):
        super().__init__()
        self.ser = ser
        self.data_queue = data_queue
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            try:
                raw_data = self.ser.readline()
                decoded_data = raw_data.decode("utf-8").strip()

                if ": " in decoded_data:
                    data_str = decoded_data.split(": ")[1]
                    data_values = data_str.split(",")
                    if len(data_values) == 6:
                        data = list(map(float, data_values))
                        self.data_queue.put(data)
                    else:
                        print(f"Unexpected data length: {len(data_values)}")
                else:
                    print(f"Unexpected data format: {decoded_data}")

            except Exception as e:
                print(f"Error reading from serial port: {e}")

    def stop(self):
        self.stop_event.set()
