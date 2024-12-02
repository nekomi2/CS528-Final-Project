import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_live(recording_manager, data_queue):
    max_points = 100
    accel_data = np.zeros((3, max_points))
    gyro_data = np.zeros((3, max_points))
    data_index = 0

    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle("MPU6050 Live Data from Serial Port", fontsize=16)

    accel_labels = ["Accel X", "Accel Y", "Accel Z"]
    gyro_labels = ["Gyro X", "Gyro Y", "Gyro Z"]
    y_labels = ["Acceleration (g)", "Angular Velocity (deg/s)"]

    accel_colors = ["r", "g", "b"]
    gyro_colors = ["c", "m", "y"]

    accel_lines = [
        axs[0].plot([], [], color, label=label)[0]
        for color, label in zip(accel_colors, accel_labels)
    ]
    gyro_lines = [
        axs[1].plot([], [], color, label=label)[0]
        for color, label in zip(gyro_colors, gyro_labels)
    ]

    axs[0].set_ylim(-4, 4)
    axs[0].set_xlim(0, max_points - 1)
    axs[0].set_title("Accelerometer Data", fontsize=14)
    axs[0].set_ylabel(y_labels[0], fontsize=12)
    axs[0].legend(loc="upper right")
    axs[0].grid(True)

    axs[1].set_ylim(-500, 500)
    axs[1].set_xlim(0, max_points - 1)
    axs[1].set_title("Gyroscope Data", fontsize=14)
    axs[1].set_ylabel(y_labels[1], fontsize=12)
    axs[1].set_xlabel("Samples", fontsize=12)
    axs[1].legend(loc="upper right")
    axs[1].grid(True)

    status_text = fig.text(0.02, 0.95, "", fontsize=12, color="green")

    def update_plot(frame):
        nonlocal data_index
        while not data_queue.empty():
            data = data_queue.get()
            if len(data) == 6:
                accel_data[:, data_index] = data[:3]
                gyro_data[:, data_index] = data[3:]

                recording_manager.write_data(data)

                data_index = (data_index + 1) % max_points

        x_data = np.arange(max_points)

        for i, line in enumerate(accel_lines):
            shifted_data = np.roll(accel_data[i], -data_index)
            line.set_data(x_data, shifted_data)

        for i, line in enumerate(gyro_lines):
            shifted_data = np.roll(gyro_data[i], -data_index)
            line.set_data(x_data, shifted_data)

        is_recording, countdown = recording_manager.get_status()
        status_text.set_text(
            f"Recording... {countdown}s left"
            if is_recording
            else f"Paused... {countdown}s left"
        )
        status_text.set_color("green" if is_recording else "red")

        return accel_lines + gyro_lines + [status_text]

    ani = FuncAnimation(fig, update_plot, interval=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
