from djitellopy import Tello
import time
import cv2


class TelloDroneBuilder:
    def __init__(self):
        self.drone = Tello()
        self.drone.connect()
        print(f"Battery level: {self.drone.get_battery()}%")

    def take_off(self):
        print("Taking off...")
        self.drone.takeoff()
        return self

    def land(self):
        print("Landing...")
        self.drone.land()
        return self

    def set_speed(self, speed):
        print(f"Setting speed to {speed} cm/s.")
        self.drone.set_speed(speed)
        return self

    def set_velocity(self, vx, vy, vz, yaw):
        """
        Set the drone's velocity in the x, y, z directions and yaw.

        Args:
            vx (int): Velocity in the x-direction (forward/backward).
            vy (int): Velocity in the y-direction (left/right).
            vz (int): Velocity in the z-direction (up/down).
            yaw (int): Rotational velocity (yaw rate).
        """
        print(f"Setting velocity: vx={vx}, vy={vy}, vz={vz}, yaw={yaw}")
        self.drone.send_rc_control(vx, vy, vz, yaw)
        return self

    def move_forward(self, distance):
        print(f"Moving forward {distance} cm.")
        self.drone.move_forward(distance)
        return self

    def move_up(self, distance):
        print(f"Moving up {distance} cm.")
        self.drone.move_up(distance)
        return self

    def move_down(self, distance):
        print(f"Moving down {distance} cm.")
        self.drone.move_down(distance)
        return self

    def move_back(self, distance):
        print(f"Moving back {distance} cm.")
        self.drone.move_back(distance)
        return self

    def move_left(self, distance):
        print(f"Moving left {distance} cm.")
        self.drone.move_left(distance)
        return self

    def move_right(self, distance):
        print(f"Moving right {distance} cm.")
        self.drone.move_right(distance)
        return self

    def rotate(self, angle):
        print(f"Rotating by {angle} degrees.")
        if angle > 0:
            self.drone.rotate_clockwise(angle)
        else:
            self.drone.rotate_counter_clockwise(abs(angle))
        return self

    def flip_back(self):
        self.drone.flip_back()
        return self

    def capture_image(self, filename="capture.jpg"):
        print(f"Capturing image: {filename}")
        frame = self.drone.get_frame_read().frame
        cv2.imwrite(filename, frame)
        return self

    def record_video(self, duration, filename="video.avi"):
        print(f"Recording video for {duration} seconds.")
        frame_read = self.drone.get_frame_read()
        height, width, _ = frame_read.frame.shape
        video_writer = cv2.VideoWriter(
            filename, cv2.VideoWriter_fourcc(*"XVID"), 30, (width, height)
        )

        start_time = time.time()
        while time.time() - start_time < duration:
            frame = frame_read.frame
            video_writer.write(frame)
        video_writer.release()
        print(f"Video saved as {filename}")
        return self

    def fly_square(self, distance):
        print(f"Flying in a square pattern with {distance} cm sides.")
        for _ in range(4):
            self.move_forward(distance)
            self.rotate(90)
        return self

    def fly_to(self, x, y, z, speed=20):
        print(f"Flying to coordinates: x={x}, y={y}, z={z} at speed {speed}.")
        self.drone.go_xyz_speed(x, y, z, speed)
        return self

    def emergency_stop(self):
        print("Emergency stop!")
        self.drone.emergency()
        return self

    def disconnect(self):
        print("Disconnecting from drone.")
        self.drone.end()
        return self
