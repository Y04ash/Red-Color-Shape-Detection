import cv2
import numpy as np

def nothing(x):
    pass

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create trackbars for HSV range adjustment
cv2.namedWindow('Trackbars')
cv2.createTrackbar('L-H', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('L-S', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('L-V', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('U-H', 'Trackbars', 179, 179, nothing)
cv2.createTrackbar('U-S', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('U-V', 'Trackbars', 255, 255, nothing)

while True:
    # ret, frame = cap.read()
    # if not ret:
    #     break
    frame = cv2.imread('C:\\Users\\DELL\\Desktop\\yash\\python\\cv\\ObjectMeasurement\\cards.jpg')
    frame = cv2.resize(frame, (0,0),fx=0.5,fy=0.5)
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get HSV values from trackbars
    l_h = cv2.getTrackbarPos('L-H', 'Trackbars')
    l_s = cv2.getTrackbarPos('L-S', 'Trackbars')
    l_v = cv2.getTrackbarPos('L-V', 'Trackbars')
    u_h = cv2.getTrackbarPos('U-H', 'Trackbars')
    u_s = cv2.getTrackbarPos('U-S', 'Trackbars')
    u_v = cv2.getTrackbarPos('U-V', 'Trackbars')

    # Define lower and upper HSV bounds
    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])

    # Create mask based on HSV range
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Morphological operations to clean the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cont in contours:
        area = cv2.contourArea(cont)
        if area > 500:  # Filter out small shapes
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(cont, True)
            approx = cv2.approxPolyDP(cont, epsilon, True)

            # Draw the contour and shape
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)

            # Identify the shape
            corners = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            if corners == 3:
                shape_name = "Triangle"
            elif corners == 4:
                aspect_ratio = w / float(h)
                shape_name = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
            elif corners == 5:
                shape_name = "Pentagon"
            elif corners > 5:
                shape_name = "Circle"
            else:
                shape_name = "Unknown"

            # Put the shape name near the detected shape
            cv2.putText(frame, shape_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Show the results
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
