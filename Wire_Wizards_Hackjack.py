from ultralytics import YOLO
import cv2
import time  # Import time module

# List of card names corresponding to class IDs
card_names = [
    "10C", "10D", "10H", "10S", "2C", "2D", "2H", "2S", "3C", "3D", "3H", "3S",
    "4C", "4D", "4H", "4S", "5C", "5D", "5H", "5S", "6C", "6D", "6H", "6S", 
    "7C", "7D", "7H", "7S", "8C", "8D", "8H", "8S", "9C", "9D", "9H", "9S", 
    "AC", "AD", "AH", "AS", "BACK", "BLACK JOKER", "JC", "JD", "JH", "JS", 
    "KC", "KD", "KH", "KS", "QC", "QD", "QH", "QS", "RED JOKER"
]

# Dictionary for card values (Ace as 'A', face cards are 10, others are their numeric value)
card_values = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "10": 10, "J": 10, "Q": 10, "K": 10, "A": "A"  # Ace is represented as 'A'
}

# Load the trained YOLOv8 model
model_path = "C:\\Users\\Jugal pahuja\\OneDrive\\Desktop\\best (1).pt"  # Correct path to your model on Windows
model = YOLO(model_path)

# Initialize the video capture (0 for webcam or provide a file path for a video)
cap = cv2.VideoCapture(0)

# Dictionary to store detected cards with high confidence
detected_cards = {"dealer": [], "player": []}  # Store for both dealer and player separately

# Variables to track time gap for card detection (last detected time)
last_detection_time = time.time()  # Initial time to track the time gap
time_gap = 3  # 3 seconds gap between storing cards

# Helper function to calculate the hand's total value
def calculate_hand_value(cards):
    total_value = 0
    ace_count = 0

    for card, value in cards:
        if value == "A":
            ace_count += 1
            total_value += 11  # Consider Ace as 11 initially
        else:
            total_value += value

    # Adjust Ace value if it causes a bust
    while total_value > 21 and ace_count > 0:
        total_value -= 10  # Convert Ace from 11 to 1
        ace_count -= 1

    return total_value

# Function to make a prediction whether to hit or pass based on the player's hand
def predict_action(player_sum, dealer_first_card_value):
    # Check if the player has exceeded 21 (bust)
    if player_sum > 21:
        return "Player Loses"
    elif player_sum < 17:  # Simple rule for hit/pass
        return "Hit"
    else:
        return "Pass"

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)

    # Extract predictions (boxes, labels, and confidences)
    boxes = results[0].boxes.xywh.cpu().numpy()  # Bounding boxes (xywh)
    labels = results[0].boxes.cls.cpu().numpy()  # Class labels
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores

    # Get the dimensions of the frame
    height, width, _ = frame.shape

    # Draw a dividing line for the dealer (top) and player (bottom)
    cv2.line(frame, (0, height // 2), (width, height // 2), (0, 255, 0), 2)  # Green line

    # Add the text "Dealer" on the upper half
    cv2.putText(frame, "Dealer", (width // 2 - 50, height // 4), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2, cv2.LINE_AA)

    # Add the text "Player" on the lower half
    cv2.putText(frame, "Player", (width // 2 - 50, 3 * height // 4), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2, cv2.LINE_AA)

    # Annotate the frame with bounding boxes and card names
    for box, label, confidence in zip(boxes, labels, confidences):
        x1, y1, w, h = box
        x2, y2 = int(x1 + w), int(y1 + h)
        
        # Get the card name from the list using the label as the index
        card_name = card_names[int(label)]
        card_value = card_name[:-1]  # Remove the suit to get the rank (e.g., '10', 'A', 'K')
        stored_value = card_values.get(card_value, card_value)  # Get value from card_values dict
        
        # Get the current time
        current_time = time.time()

        # If 3 seconds have passed, store the card and update the last detection time
        if current_time - last_detection_time >= time_gap:
            # If the card is in the dealer's area (top half), store it in dealer's list
            if y1 < height // 2:
                if card_name not in [card[0] for card in detected_cards["dealer"]]:
                    detected_cards["dealer"].append((card_name, stored_value))
            
            # If the card is in the player's area (bottom half), store it in player's list
            else:
                if card_name not in [card[0] for card in detected_cards["player"]]:
                    detected_cards["player"].append((card_name, stored_value))
            
            # Update the last detection time
            last_detection_time = current_time

        # Draw bounding boxes and label with card name and value
        cv2.rectangle(frame, (int(x1), int(y1)), (x2, int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{card_name} ({stored_value}) {confidence:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Calculate dealer's and player's hand values
    player_sum = calculate_hand_value(detected_cards["player"])
    dealer_sum = calculate_hand_value(detected_cards["dealer"])

    # Predict whether to hit or pass based on the player's hand value
    action = predict_action(player_sum, dealer_sum)

    # Display the dealer's cards and sum
    dealer_text = f"Dealer's Cards: {', '.join([f'{c[0]}({c[1]})' for c in detected_cards['dealer']])}"
    cv2.putText(frame, dealer_text, (10, height // 5 + 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the player's cards and sum
    player_text = f"Player's Cards: {', '.join([f'{c[0]}({c[1]})' for c in detected_cards['player']])} | Total: {player_sum}"
    cv2.putText(frame, player_text, (10, 3 * height // 5 + 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the suggested action
    cv2.putText(frame, f"Suggested action: {action}", (width // 2 - 100, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the processed frame
    cv2.imshow('YOLOv8 Live Detection', frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()