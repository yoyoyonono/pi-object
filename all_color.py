import cv2
from jeevan import get_limits
from PIL import Image   # Python Imaging Library
from dataclasses import dataclass
import numpy as np

@dataclass
class ColorObject:
    id: int
    bbox: tuple
    score: int = 0.3

# BGR values for different colors
red = [0, 0, 255]
blue = [255, 0, 0]
green = [0, 255, 0]
yellow = [0, 255, 255]

zoom = 0.0

def identify_colors(frame):
    # Convert the frame to HSV color space
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for each color
    lower_red, upper_red = get_limits(red)
    lower_blue, upper_blue = get_limits(blue)
    lower_green, upper_green = get_limits(green)
    lower_yellow, upper_yellow = get_limits(yellow)
    
    mask_red = cv2.inRange(hsvImage, lower_red, upper_red)
    mask_blue = cv2.inRange(hsvImage, lower_blue, upper_blue)
    mask_green = cv2.inRange(hsvImage, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsvImage, lower_yellow, upper_yellow)

    #  Define a kernel for the morphological operation
    kernel = np.ones((10, 10), np.uint8)

    # Apply opening to each mask to remove noise
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)

    # Define threshold for left, right and center
    center = frame.shape[1] // 2        # Center of the frame\
    threshold = frame.shape[1] * 0.05     # 10% of the frame width
    
    left_threshold = center - threshold         
    right_threshold = center + threshold

    # Find centroids and positions for each color
    centroids_positions = {}
    for mask, color in [(mask_red, 'red'), (mask_blue, 'blue'), (mask_green, 'green'), (mask_yellow, 'yellow')]:
        mask_ = Image.fromarray(mask)
        bbox = mask_.getbbox()  # returns bounding box for the mask

        if bbox is not None:
            x1, y1, x2, y2 = bbox

            # Calculate the centroid
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Determine if the object is to the left or right of the center
            if cx < left_threshold:
                position = 'left'
            elif cx > right_threshold:
                position = 'right'
            else:
                position = 'center'

            centroids_positions[color] = ((cx, cy), position)

    return centroids_positions, {'red': mask_red, 'blue': mask_blue, 'green': mask_green, 'yellow': mask_yellow}

def find_objects(frame):
    # Convert the frame to HSV color space
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for each color
    lower_red, upper_red = get_limits(red)
    lower_blue, upper_blue = get_limits(blue)
    lower_green, upper_green = get_limits(green)
    lower_yellow, upper_yellow = get_limits(yellow)
    
    mask_red = cv2.inRange(hsvImage, lower_red, upper_red)
    mask_blue = cv2.inRange(hsvImage, lower_blue, upper_blue)
    mask_green = cv2.inRange(hsvImage, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsvImage, lower_yellow, upper_yellow)

    #  Define a kernel for the morphological operation
    kernel = np.ones((10, 10), np.uint8)

    # Apply opening to each mask to remove noise
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)

    # Define threshold for left, right and center
    center = frame.shape[1] // 2        # Center of the frame\
    threshold = frame.shape[1] * 0.05     # 10% of the frame width
    
    left_threshold = center - threshold         
    right_threshold = center + threshold

    # Find centroids and positions for each color

    color_objects = []

    for mask, color in [(mask_red, 1), (mask_blue, 2), (mask_green, 3), (mask_yellow, 4)]:
        mask_ = Image.fromarray(mask)
        bbox = mask_.getbbox()  # returns bounding box for the mask
        if bbox is not None and color == 3:
            bbox_real = (bbox[0] / frame.shape[1], bbox[1] / frame.shape[0], bbox[2] / frame.shape[1], bbox[3] / frame.shape[0])
            color_objects.append(ColorObject(id=color, bbox=bbox_real))

    return color_objects

def main():
    # Open the camera
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()

        centroids_positions, masks = identify_colors(frame)

        # shapes = {}
        # for color, mask in masks.items():
        #     shape = identify_shapes(mask)
        #     shapes[color] = shape
    
        # # Print the shapes in terminal
        # print(shapes)

        # Draw the centroids and positions on the frame
        for color, ((cx, cy), position) in centroids_positions.items():
            frame = cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(frame, f'{color} {position}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if not ret:
            print("Can't receive frame")
            break

        # Resize the frame
        frame = cv2.resize(frame, (800, 600))  # You can adjust the values as needed

        cv2.imshow('real', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()















# def identify_shapes(mask):
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for cnt in contours:
#         # Get approximate polygon
#         epsilon = 0.02 * cv2.arcLength(cnt, True)
#         approx = cv2.approxPolyDP(cnt, epsilon, True)

#         if len(approx) == 3:
#             shape = "triangle"  # Could be a cone viewed from the side
#         elif len(approx) == 4:
#             shape = "rectangle"  # Could be a cube or cylinder
#         else:
#             # Check if shape is a circle
#             (x, y), radius = cv2.minEnclosingCircle(cnt)
#             center = (int(x), int(y))
#             radius = int(radius)
#             if abs(cv2.contourArea(cnt) - np.pi * radius**2) < 100:
#                 shape = "circle"  # Could be a sphere or cone viewed from the top
#             else:
#                 shape = "unknown"

#         return shape
    









# import cv2
# from jeevan import get_limits
# from PIL import Image   # Python Imaging Library

# # BGR values for different colors
# red = [0, 0, 255]
# blue = [255, 0, 0]
# green = [0, 255, 0]
# yellow = [0, 255, 255]

# zoom = 0.0

# # Open the camera
# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()

#     # Crop the frame
#     height, width = frame.shape[:2]
#     start_row, start_col = int(height * zoom), int(width * zoom)
#     end_row, end_col = int(height * (1-zoom)), int(width * (1-zoom))
#     frame = frame[start_row:end_row, start_col:end_col]

#     # Convert the frame to HSV color space
#     hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # Create masks for each color
#     lower_red, upper_red = get_limits(red)
#     lower_blue, upper_blue = get_limits(blue)
#     lower_green, upper_green = get_limits(green)
#     lower_yellow, upper_yellow = get_limits(yellow)
    
#     mask_red = cv2.inRange(hsvImage, lower_red, upper_red)
#     mask_blue = cv2.inRange(hsvImage, lower_blue, upper_blue)
#     mask_green = cv2.inRange(hsvImage, lower_green, upper_green)
#     mask_yellow = cv2.inRange(hsvImage, lower_yellow, upper_yellow)

#     # Find bounding boxes for each color
#     for mask, color in [(mask_red, 'red'), (mask_blue, 'blue'), (mask_green, 'green'), (mask_yellow, 'yellow')]:
#         mask_ = Image.fromarray(mask)
#         bbox = mask_.getbbox()  # returns bounding box for the mask

#         if bbox is not None:
#             x1, y1, x2, y2 = bbox
#             frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
#             cv2.putText(frame, color, (x1 + (x2-x1)//2, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#     if not ret:
#         print("Can't receive frame")
#         break

#     # Resize the frame
#     frame = cv2.resize(frame, (800, 600))  # You can adjust the values as needed

#     cv2.imshow('real', frame)
#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




























# import cv2
# import numpy as np

# def detect_color(image):
#     # Convert image to HSV color space
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#     # Define color ranges
#     lower_blue = np.array([100, 50, 50])
#     upper_blue = np.array([130, 255, 255])
#     lower_red = np.array([0, 50, 50])
#     upper_red = np.array([10, 255, 255])
#     lower_green = np.array([40, 50, 50])
#     upper_green = np.array([80, 255, 255])
#     lower_yellow = np.array([20, 50, 50])
#     upper_yellow = np.array([40, 255, 255])
#     lower_black = np.array([0, 0, 0])
#     upper_black = np.array([180, 255, 30])
#     lower_white = np.array([0, 0, 200])
#     upper_white = np.array([180, 30, 255])

#     # Threshold the image to get only the specified colors
#     blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
#     red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
#     green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
#     yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
#     black_mask = cv2.inRange(hsv_image, lower_black, upper_black)
#     white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

#     # Bitwise-AND the masks with the original image
#     blue_result = cv2.bitwise_and(image, image, mask=blue_mask)
#     red_result = cv2.bitwise_and(image, image, mask=red_mask)
#     green_result = cv2.bitwise_and(image, image, mask=green_mask)
#     yellow_result = cv2.bitwise_and(image, image, mask=yellow_mask)
#     black_result = cv2.bitwise_and(image, image, mask=black_mask)
#     white_result = cv2.bitwise_and(image, image, mask=white_mask)

#     # Display the results
#     cv2.imshow('Blue', blue_result)
#     cv2.imshow('Red', red_result)
#     cv2.imshow('Green', green_result)
#     cv2.imshow('Yellow', yellow_result)
#     cv2.imshow('Black', black_result)
#     cv2.imshow('White', white_result)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # Load the image
# image = cv2.imread('image.jpg')

# # Call the color detection function
# detect_color(image)