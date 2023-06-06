import cv2
import numpy as np
import struct
import serial
import time

cap = cv2.VideoCapture(0)
l1 = 8  # link 1
l2 = 8  # link 2

ser = serial.Serial('COM6', baudrate=9600, bytesize=8, timeout=2)
fmt = 'iii'

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture frame from camera")
        break

    frame_height, frame_width, _ = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([15, 255, 255])

    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if radius > 10:
            ball_x = int(x - frame_width / 2)
            ball_y = int(frame_height - y)

            cos_theta2 = (ball_x ** 2 + ball_y ** 2 - l1 **
                          2 - l2 ** 2) / (2 * l1 * l2)
            sin_theta2 = np.sqrt(1 - cos_theta2 ** 2)
            theta2 = np.arctan2(sin_theta2, cos_theta2)

            k1 = l1 + l2 * cos_theta2
            k2 = l2 * sin_theta2

            theta1 = np.arctan2(ball_y, ball_x) - np.arctan2(k2, k1)
            servo3 = 0

            if np.isnan(theta1) and np.isnan(theta2):
                theta1 = np.arctan2(y, x)
                theta2 = 0
                servo3 = 1

            theta1_deg = np.rad2deg(theta1)
            theta2_deg = np.rad2deg(theta2)

            print(f"theta1 = {int(theta1_deg)} degrees")
            print(f"theta2 = {int(theta2_deg)} degrees")

            packet = struct.pack(fmt, int(theta1_deg), int(theta2_deg), servo3)
            ser.write(packet)
            ser.flush()
            time.sleep(.1)

            cv2.circle(frame, (int(frame_width / 2 + ball_x), int(frame_height - ball_y)), int(radius),
                       (0, 255, 255), 2)
            cv2.putText(frame, "x: {}, y: {}".format(ball_x, ball_y),
                        (int(frame_width / 2 + ball_x) - 50,
                         int(frame_height - ball_y) - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, "theta1:{}".format(theta1_deg), (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, "theta2:{}".format(theta2_deg), (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
