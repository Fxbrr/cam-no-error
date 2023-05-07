import cv2
import numpy as np
import struct
# import serial


cap = cv2.VideoCapture(0)

focal_length = 755.81  # Update this value with your camera's focal length
ball_diameter = 4.3  # Update this value with the diameter of your ball in cm
l1 = 2  # link 1
l2 = 2.50  # link 2

L1 = l1*37.7952755906  # link 1 in px
L2 = l2*37.7952755906  # link 2 in px


# open serial comms
# ser = serial.Serial('/dev/ttyACM0', 9600)
fmt = 'ff'

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_height, frame_width, _ = frame.shape

    if not ret:
        # Handle case when camera is not working
        print("Error: Could not capture frame from camera")
        break

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds of orange color in HSV
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([15, 255, 255])

    # Threshold the HSV image to get only orange colors
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Apply a series of morphological transformations to remove noise and fill in gaps
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if a contour was found
    if len(contours) > 0:
        # Find the largest contour in the mask
        c = max(contours, key=cv2.contourArea)

        # Get the (x, y) coordinates and radius of the ball
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        # Only proceed if the radius meets a minimum size
        if radius > 10:
            # Calculate the distance from the camera to the ball
            distance = (focal_length * ball_diameter) / (2 * radius)

            # assign ball_x ball_y
            ball_x = x - frame_width // 2
            ball_y = frame_height - y

            # Calculate theta2 using the law of cosines
            cos_theta2 = (ball_x**2 + ball_y**2 - L1**2 - L2**2) / (2*L1*L2)
            sin_theta2 = np.sqrt(1 - cos_theta2**2)
            theta2 = np.arctan2(sin_theta2, cos_theta2)

            # Calculate k1 and k2
            k1 = L1 + L2*cos_theta2
            k2 = L2*sin_theta2

            # Calculate theta1 using inverse tangent function
            theta1 = np.arctan2(ball_y, ball_x) - np.arctan2(k2, k1)

            # Convert to degrees
            float
            theta1_deg = np.rad2deg(theta1)
            float
            theta2_deg = np.rad2deg(theta2)

            # Print the joint angles
            print(f"theta1 = {theta1_deg:.2f} degrees")
            print(f"theta2 = {theta2_deg:.2f} degrees")
            packet = struct.pack(fmt, theta1_deg, theta2_deg)
            # ser.write(packet)

            # Draw the circle and ball's coordinates on the frame
            cv2.circle(frame, (int(frame_width // 2 + ball_x),
                               int(frame_height - ball_y)), int(radius), (0, 255, 255), 2)
            cv2.putText(frame, "x: {}, y: {}".format(int(ball_x), int(ball_y)), (int(frame_width // 2 + ball_x) - 50, int(frame_height - ball_y) - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Wait for 'q' key to be pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
