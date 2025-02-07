# import cv2
# import mediapipe as mp
# import numpy as np

# # Initialize Mediapipe Pose Estimation
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()

# # Open Video Capture
# cap = cv2.VideoCapture(0)

# # Check if the camera is opened
# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()

# # Store previous landmarks
# previous_landmarks = None
# motion_threshold = 60  # Adjust sensitivity

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Could not read frame.")
#         break

#     # Convert BGR to RGB for Mediapipe
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(rgb_frame)

#     if results.pose_landmarks:
#         # Extract the landmark positions
#         current_landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in results.pose_landmarks.landmark]
        
#         # Draw pose landmarks on the frame
#         mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#         # Debugging: Print landmarks
#         print("Detected Landmarks:", current_landmarks)

#         # Check motion if previous landmarks exist
#         if previous_landmarks:
#             motion_detected = False
#             for i in range(len(current_landmarks)):
#                 x1, y1 = previous_landmarks[i]
#                 x2, y2 = current_landmarks[i]
                
#                 # Compute Euclidean distance (scaled to pixel values)
#                 distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

#                 # If movement exceeds threshold, flag motion
#                 if distance > motion_threshold:
#                     motion_detected = True
#                     break  # No need to check further if motion is detected

#             if motion_detected:
#                 cv2.putText(frame, "MOTION DETECTED", (50, 50), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

#         # Update previous landmarks
#         previous_landmarks = current_landmarks

#     else:
#         print("No person detected!")

#     # Show the video feed
#     cv2.imshow("Motion Detection", frame)

#     # Exit on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()


# python "c:\Users\durba\Downloads\OpenCV_Tutorial\electro.py"

# steps to run
## win+x
## Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
##.\Downloads\OpenCV_Tutorial\mediapipe_env\Scripts\Activate.ps1
## pip install mediapipe
## python "c:\Users\durba\Downloads\OpenCV_Tutorial\electro.py"






import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

# pipeline = rs.pipeline()
# config = rs.config()

# # Get device product line for setting a supporting resolution
# pipeline_wrapper = rs.pipeline_wrapper(pipeline)
# pipeline_profile = config.resolve(pipeline_wrapper)
# device = pipeline_profile.get_device()
# device_product_line = str(device.get_info(rs.camera_info.product_line))

# found_rgb = False
# for s in device.sensors:
#     if s.get_info(rs.camera_info.name) == 'RGB Camera':
#         found_rgb = True
#         break
# if not found_rgb:
#     print("The demo requires Depth camera with Color sensor")
#     exit(0)

# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# # Start streaming
# pipeline.start(config)


H = np.array([[1.2, 0.02, -30],
            [0.01, 1.1, -20],
            [0.0002, 0.0005, 1]])

motion_threshold = 60
prev_landmarks = None

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_green = np.array([35, 40, 40])
    high_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv_frame, low_green, high_green)

    kernel = np.ones((5,5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    motion_detected = False
    moving_participant_pixel_cords = []

    if results.pose_landmarks:
        current_landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in results.pose_landmarks.landmark]

        green_person = False
        for(x,y) in current_landmarks:
            if 0 <= x < green_mask.shape[1] and 0 <= y < green_mask.shape[0]:
                if green_mask[y,x]>0:
                    green_person = True
                    break
        
        if green_person:
            if prev_landmarks:
                for i in range(len(current_landmarks)):
                    x1, y1 = prev_landmarks[i]
                    x2, y2 = current_landmarks[i]

                    distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)

                    if distance > motion_threshold:
                        motion_detected = True
                        moving_participant_pixel_cords.append((x2,y2))
                        break

            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            prev_landmarks = current_landmarks

    if motion_detected:
        moving_participant_real_cords = []

        for (u, v) in moving_participant_pixel_cords:
            # Convert pixel to real-world using Homography
            pixel_point = np.array([[u, v, 1]], dtype=np.float32).T
            real_world_point = np.dot(H, pixel_point)

            # Normalize by W
            X_p = real_world_point[0] / real_world_point[2]
            Y_p = real_world_point[1] / real_world_point[2]
            Z_p = 1.7  # Assuming average human height (in meters)

            moving_participant_real_cords.append((X_p, Y_p, Z_p))

        data = str(moving_participant_real_cords).encode()
        # sock.sendto(data, (UDP_IP, UDP_PORT))
        print("Sent:", moving_participant_real_cords)


    cv2.imshow("Motion Detection - Only Green Clothes", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






# import cv2
# import numpy as np
# import mediapipe as mp
# import serial
# import time

# arduino = serial.Serial(port="COM5", baudrate=9600, timeout=1)
# time.sleep(2) 

# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()
# disqualified_targets = []

# # Open webcam
# cap = cv2.VideoCapture(0)

# # pipeline = rs.pipeline()
# # config = rs.config()

# # # Get device product line for setting a supporting resolution
# # pipeline_wrapper = rs.pipeline_wrapper(pipeline)
# # pipeline_profile = config.resolve(pipeline_wrapper)
# # device = pipeline_profile.get_device()
# # device_product_line = str(device.get_info(rs.camera_info.product_line))

# # found_rgb = False
# # for s in device.sensors:
# #     if s.get_info(rs.camera_info.name) == 'RGB Camera':
# #         found_rgb = True
# #         break
# # if not found_rgb:
# #     print("The demo requires Depth camera with Color sensor")
# #     exit(0)

# # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# # Start streaming
# # pipeline.start(config)


# H = np.array([[1.2, 0.02, -30],
#             [0.01, 1.1, -20],
#             [0.0002, 0.0005, 1]])

# motion_threshold = 60
# prev_landmarks = None

# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret or frame is None:
#         print("Error reading frame")

#     hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     low_green = np.array([35, 40, 40])
#     high_green = np.array([90, 255, 255])
#     green_mask = cv2.inRange(hsv_frame, low_green, high_green)

#     kernel = np.ones((5,5), np.uint8)
#     green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(rgb_frame)

#     motion_detected = False
#     moving_participant_pixel_cords = []

#     if results.pose_landmarks:
#         current_landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in results.pose_landmarks.landmark]

#         green_person = False
#         for(x,y) in current_landmarks:
#             if 0 <= x < green_mask.shape[1] and 0 <= y < green_mask.shape[0]:
#                 if green_mask[y,x]>0:
#                     green_person = True
#                     break
        
#         if green_person:
#             if prev_landmarks:
#                 for i in range(len(current_landmarks)):
#                     x1, y1 = prev_landmarks[i]
#                     x2, y2 = current_landmarks[i]

#                     distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)

#                     if distance > motion_threshold:
#                         motion_detected = True
#                         moving_participant_pixel_cords.append((x2,y2))
#                         break

#             mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#             prev_landmarks = current_landmarks

#     if motion_detected:
#         moving_participant_real_cords = []

#         for (u, v) in moving_participant_pixel_cords:
#             # Convert pixel to real-world using Homography
#             pixel_point = np.array([[u, v, 1]], dtype=np.float32).T
#             real_world_point = np.dot(H, pixel_point)

#             # Normalize by W
#             X_p = real_world_point[0] / real_world_point[2]
#             Y_p = real_world_point[1] / real_world_point[2]
#             Z_p = 1.7  # Assuming average human height (in meters)

#             # moving_participant_real_cords.append((X_p, Y_p, Z_p))

#             # Check if this target has already been disqualified
#             if any(np.linalg.norm(np.array([X_p, Y_p]) - np.array(t)) < 1 for t in disqualified_targets):
#                 continue  # Skip sending this target

#             # Send Real-World Coordinates to Arduino
#             data = f"{float(X_p):.2f},{float(Y_p):.2f},{float(Z_p):.2f}\n".encode()

#             arduino.write(data)
#             print("Sent to Arduino:", data.decode())

#             # Mark this person as disqualified
#             disqualified_targets.append((X_p, Y_p))
#             break  # Send only one unique target at a time

#         # Send Real-World Coordinates to Python Server via UDP
#         # data = str(moving_participant_real_cords).encode()
#         # # sock.sendto(data, (UDP_IP, UDP_PORT))
#         # print("Sent:", moving_participant_real_cords)

#     # Display video
#     cv2.imshow("Motion Detection - Only Green Clothes", frame)

#     # Exit on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

# while True:
#     res, frame = cap.read()
#     if not res:
#         print("Error reading the video")
#         break
    
#     # Convert to HSV color space
#     hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # Define HSV range for green (tune these values as needed)
#     low_green = np.array([35, 40, 40])   # Lower bound of green
#     high_green = np.array([90, 255, 255]) # Upper bound of green

#     # Create mask
#     green_mask = cv2.inRange(hsv_frame, low_green, high_green)

#     # Remove small noise using morphological operations
#     kernel = np.ones((5, 5), np.uint8)
#     green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
#     green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

#     # Find contours
#     contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if area > 500:  # Filter small contours (adjust threshold as needed)
#             x, y, w, h = cv2.boundingRect(contour)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             # cv2.putText(frame, "Green Clothing Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        


#     # Display results
#     # cv2.imshow("Original Frame", frame)
#     # cv2.imshow("Green Mask", green_mask)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# Release resources
# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import mediapipe as mp
# import serial
# import time

# try:
#     arduino = serial.Serial(port="COM5", baudrate=9600, timeout=1)
#     time.sleep(2)  # Wait for the connection to initialize

#     # Your code to communicate with Arduino
#     # arduino.write(b"Hello Arduino\n")

# finally:
#     if 'arduino' in locals() or 'arduino' in globals():
#         arduino.close()  # Ensure the port is closed

# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()
# disqualified_targets = []

# # Open webcam
# cap = cv2.VideoCapture(0)

# H = np.array([[1.2, 0.02, -30],
#             [0.01, 1.1, -20],
#             [0.0002, 0.0005, 1]])

# motion_threshold = 60
# prev_landmarks = None

# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret or frame is None:
#         print("Error reading frame")

#     hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     low_green = np.array([35, 40, 40])
#     high_green = np.array([90, 255, 255])
#     green_mask = cv2.inRange(hsv_frame, low_green, high_green)

#     kernel = np.ones((5,5), np.uint8)
#     green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(rgb_frame)

#     motion_detected = False
#     moving_participant_pixel_cords = []

#     if results.pose_landmarks:
#         current_landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in results.pose_landmarks.landmark]

#         green_person = False
#         for(x,y) in current_landmarks:
#             if 0 <= x < green_mask.shape[1] and 0 <= y < green_mask.shape[0]:
#                 if green_mask[y,x]>0:
#                     green_person = True
#                     break
        
#         if green_person:
#             if prev_landmarks:
#                 for i in range(len(current_landmarks)):
#                     x1, y1 = prev_landmarks[i]
#                     x2, y2 = current_landmarks[i]

#                     distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)

#                     if distance > motion_threshold:
#                         motion_detected = True
#                         moving_participant_pixel_cords.append((x2,y2))
#                         break

#             mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#             prev_landmarks = current_landmarks

#     if motion_detected:
#         for (u, v) in moving_participant_pixel_cords:
#             # Convert pixel to real-world using Homography
#             pixel_point = np.array([[u, v, 1]], dtype=np.float32).T
#             real_world_point = np.dot(H, pixel_point)

#             # Normalize by W
#             X_p = real_world_point[0] / real_world_point[2]
#             Y_p = real_world_point[1] / real_world_point[2]
#             Z_p = 1.7  # Assuming average human height (in meters)

#             # Check if this target has already been disqualified
#             if any(np.linalg.norm(np.array([X_p, Y_p]) - np.array(t)) < 1 for t in disqualified_targets):
#                 continue  # Skip sending this target

#             # Send Real-World Coordinates to Arduino
#             data = f"{float(X_p):.2f},{float(Y_p):.2f},{float(Z_p):.2f}\n".encode()
#             arduino.write(data)
#             print("Sent to Arduino:", data.decode())

#             # Wait for Arduino to process the target
#             time.sleep(5)  # Adjust this delay based on the time it takes for the Arduino to process the target

#             # Mark this person as disqualified
#             disqualified_targets.append((X_p, Y_p))
#             break  # Send only one unique target at a time

#     # Display video
#     cv2.imshow("Motion Detection - Only Green Clothes", frame)

#     # Exit on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()





# import cv2
# import numpy as np
# import mediapipe as mp
# import serial
# import time

# # Initialize ESP32-CAM (assuming integration)
# # Connect to WiFi (if needed)

# # Initialize Mediapipe Pose Model
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()

# # Load Homography Matrix H (Precomputed)
# H = np.array([[1.2, 0.02, -30],
#               [0.01, 1.1, -20],
#               [0.0002, 0.0005, 1]])

# # Set motion threshold for movement detection
# motion_threshold = 30
# previous_landmarks = None

# # Initialize Serial Connection with Arduino
# try:
#     arduino = serial.Serial(port="COM13", baudrate=9600, timeout=1)
#     time.sleep(2)  # Allow connection to establish
# except serial.SerialException as e:
#     print("Error: Could not open serial port", e)
#     arduino = None

# # Open webcam
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()

# while True:
#     # Capture frame from camera
#     ret, frame = cap.read()
#     if not ret or frame is None:
#         print("Error reading frame")
#         break

#     # Convert frame to RGB and HSV formats
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # Apply green color mask
#     low_green = np.array([35, 40, 40])
#     high_green = np.array([90, 255, 255])
#     green_mask = cv2.inRange(hsv_frame, low_green, high_green)

#     # Perform morphological operations to reduce noise
#     kernel = np.ones((5, 5), np.uint8)
#     green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

#     # Process frame with Mediapipe to extract landmarks
#     results = pose.process(rgb_frame)

#     if results.pose_landmarks:
#         current_landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
#                              for lm in results.pose_landmarks.landmark]

#         # Filter landmarks: Keep only those within green mask
#         motion_detected = False
#         moving_participant_pixel_coords = []

#         if previous_landmarks:
#             for i in range(len(current_landmarks)):
#                 x1, y1 = previous_landmarks[i]
#                 x2, y2 = current_landmarks[i]

#                 # Compute Euclidean distance at time t
#                 D_t = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

#                 # Compute Euclidean distance at time t-1
#                 if i > 0:
#                     x1_prev, y1_prev = previous_landmarks[i - 1]
#                     x2_prev, y2_prev = current_landmarks[i - 1]
#                     D_t_minus_1 = np.sqrt((x2_prev - x1_prev) ** 2 + (y2_prev - y1_prev) ** 2)
#                 else:
#                     D_t_minus_1 = 0  # Default for the first point

#                 if abs(D_t - D_t_minus_1) > motion_threshold:
#                     motion_detected = True
#                     moving_participant_pixel_coords.append((x2, y2))
#                     break  # Only track the first detected moving person

#         previous_landmarks = current_landmarks  # Store current landmarks

#     if motion_detected:
#         moving_participant_real_coords = []
#         for (u, v) in moving_participant_pixel_coords:
#             # Compute real-world coordinates using Homography
#             pixel_point = np.array([[u, v, 1]], dtype=np.float32).T
#             real_world_point = np.dot(H, pixel_point)

#             # Normalize by W
#             Xr = real_world_point[0] / real_world_point[2]
#             Yr = real_world_point[1] / real_world_point[2]
#             Zr = 1.7  # Assume height of player

#             moving_participant_real_coords.append((Xr, Yr, Zr))

#         # Send data to Arduino or Python server
#         if arduino:
#             for coords in moving_participant_real_coords:
#                 data = f"{coords[0]:.2f},{coords[1]:.2f},{coords[2]:.2f}\n".encode()
#                 arduino.write(data)
#                 print("Sent to Arduino:", data.decode())

#     # Display the processed frame
#     cv2.imshow("Motion Detection - Green Clothes", frame)

#     # Exit condition
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()
# if arduino:
#     arduino.close()






# import cv2
# import numpy as np
# import mediapipe as mp
# import serial
# import time

# # Initialize ESP32-CAM (assuming integration)
# # Connect to WiFi (if needed)

# # Initialize Mediapipe Pose Model
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()

# # Load Homography Matrix H (Precomputed)
# H = np.array([[1.2, 0.02, -30],
#               [0.01, 1.1, -20],
#               [0.0002, 0.0005, 1]])

# # Set motion threshold for movement detection
# motion_threshold = 30
# previous_landmarks = None

# # Initialize Serial Connection with Arduino
# try:
#     arduino = serial.Serial(port="COM13", baudrate=9600, timeout=1)
#     time.sleep(2)  # Allow connection to establish
# except serial.SerialException as e:
#     print("Error: Could not open serial port", e)
#     arduino = None

# # Open webcam
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()

# while True:
#     # Capture frame from camera
#     ret, frame = cap.read()
#     if not ret or frame is None:
#         print("Error reading frame")
#         break

#     # Convert frame to RGB and HSV formats
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # Apply green color mask
#     low_green = np.array([35, 40, 40])
#     high_green = np.array([90, 255, 255])
#     green_mask = cv2.inRange(hsv_frame, low_green, high_green)

#     # Perform morphological operations to reduce noise
#     kernel = np.ones((5, 5), np.uint8)
#     green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

#     # Process frame with Mediapipe to extract landmarks
#     results = pose.process(rgb_frame)

#     if results.pose_landmarks:
#         current_landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
#                              for lm in results.pose_landmarks.landmark]

#         # Filter landmarks: Keep only those within green mask
#         motion_detected = False
#         moving_participant_pixel_coords = []

#         if previous_landmarks:
#             for i in range(len(current_landmarks)):
#                 x1, y1 = previous_landmarks[i]
#                 x2, y2 = current_landmarks[i]

#                 # Compute Euclidean distance at time t
#                 D_t = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

#                 # Compute Euclidean distance at time t-1
#                 if i > 0:
#                     x1_prev, y1_prev = previous_landmarks[i - 1]
#                     x2_prev, y2_prev = current_landmarks[i - 1]
#                     D_t_minus_1 = np.sqrt((x2_prev - x1_prev) ** 2 + (y2_prev - y1_prev) ** 2)
#                 else:
#                     D_t_minus_1 = 0  # Default for the first point

#                 if abs(D_t - D_t_minus_1) > motion_threshold:
#                     motion_detected = True
#                     moving_participant_pixel_coords.append((x2, y2))
#                     break  # Only track the first detected moving person

#         previous_landmarks = current_landmarks  # Store current landmarks

#     if motion_detected:
#         moving_participant_real_coords = []
#         for (u, v) in moving_participant_pixel_coords:
#             # Compute real-world coordinates using Homography
#             pixel_point = np.array([[u, v, 1]], dtype=np.float32).T
#             real_world_point = np.dot(H, pixel_point)

#             # Normalize by W
#             Xr = real_world_point[0] / real_world_point[2]
#             Yr = real_world_point[1] / real_world_point[2]
#             Zr = 1.7  # Assume height of player

#             moving_participant_real_coords.append((Xr, Yr, Zr))

#         # Send data to Arduino or Python server
#         if arduino:
#             for coords in moving_participant_real_coords:
#                 data = f"{coords[0]:.2f},{coords[1]:.2f},{coords[2]:.2f}\n".encode()
#                 arduino.write(data)
#                 print("Sent to Arduino:", data.decode())

#     # Display the processed frame
#     cv2.imshow("Motion Detection - Green Clothes", frame)

#     # Exit condition
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()
# if arduino:
#     arduino.close()