import cv2
import math


def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def draw_lines(image_path):
    image = cv2.imread(image_path)
    clone = image.copy()
    window_name = "Draw Lines"
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, image)

    start_point = None
    end_point = None

    def draw_line_callback(event, x, y, flags, param):
        nonlocal start_point, end_point, clone

        if event == cv2.EVENT_LBUTTONDOWN:
            start_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            end_point = (x, y)
            cv2.line(clone, start_point, end_point, (0, 0, 255), 2)
            distance = calculate_distance(
                start_point[0], start_point[1], end_point[0], end_point[1])
            cv2.putText(clone, f"{distance:.2f} pixels", (int((start_point[0] + end_point[0]) / 2), int((start_point[1] + end_point[1]) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow(window_name, clone)

    cv2.setMouseCallback(window_name, draw_line_callback)

    while True:
        cv2.imshow(window_name, clone)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            clone = image.copy()
            start_point = None
            end_point = None
        elif key == ord("q") or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


# Example usage
image_path = "/home/pi/opencv-distance/rf.png"  # Replace with your image path
draw_lines(image_path)
