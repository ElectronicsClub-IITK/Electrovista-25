import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8s_playing_cards.pt")

# Use WiFi webcam instead of a direct camera
IP_WEBCAM_URL = "http://192.168.137.59:8080/video"  # Replace with your phone's IP
cap = cv2.VideoCapture(IP_WEBCAM_URL)

# Card values for Blackjack
card_values = {
    "A": 11, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
    "J": 10, "Q": 10, "K": 10
}

total_decks = 6
used_cards = []

# Extract rank from card label
def extract_rank(card):
    return ''.join(filter(str.isalpha, card)) if card[:-1] not in card_values else card[:-1]

# Calculate Blackjack score
def calculate_score(hand):
    score, num_aces = 0, 0
    for card in hand:
        rank = extract_rank(card)
        value = card_values.get(rank, 0)
        num_aces += rank == "A"
        score += value
    while score > 21 and num_aces:
        score -= 10
        num_aces -= 1
    return score

# Blackjack strategy
def blackjack_strategy(player_score, dealer_card):
    if player_score > 21:
        return "Busted"
    dealer_value = card_values.get(extract_rank(dealer_card), 0)
    return ("Stand", "Stand", "Stand", "Double Down", "Double Down", "Double Down", "Hit")[
        (player_score < 9) + (player_score in (9, 10, 11)) * (dealer_value > 6) +
        (player_score == 12) * (dealer_value not in [4, 5, 6]) +
        (player_score >= 13) * (dealer_value >= 7)
    ]

# Probability of remaining cards
def get_card_probabilities():
    remaining_deck = {rank: total_decks * 4 for rank in card_values}
    for card in used_cards:
        rank = extract_rank(card)
        if rank in remaining_deck and remaining_deck[rank] > 0:
            remaining_deck[rank] -= 1
    remaining_total = sum(remaining_deck.values())
    return {rank: count / remaining_total for rank, count in remaining_deck.items()} if remaining_total else {}

seen_cards = {}
confirmed_player_hand, confirmed_dealer_hand = set(), set()
threshold_time = 0.05

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Camera error! Exiting...")
        break

    # Detect cards using YOLO
    results = model.predict(frame, conf=0.3, imgsz=416, verbose=False)
    detected_player_hand, detected_dealer_hand = set(), set()

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = result.names[int(box.cls[0])]

            if conf < 0.3:
                continue

            # Determine card location
            center_y = (y1 + y2) // 2
            zone = "player" if center_y > frame.shape[0] // 2 else "dealer"
            detected_hand = detected_player_hand if zone == "player" else detected_dealer_hand

            # Confirm card detection
            if label in seen_cards:
                if time.time() - seen_cards[label] > threshold_time:
                    detected_hand.add(label)
                    if label not in used_cards:
                        used_cards.append(label)
            else:
                seen_cards[label] = time.time()

            # Draw bounding box
            color = (0, 255, 0) if zone == "player" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Update confirmed hands
    confirmed_player_hand = detected_player_hand
    confirmed_dealer_hand = detected_dealer_hand

    # Calculate scores
    player_score = calculate_score(confirmed_player_hand)
    dealer_score = calculate_score(confirmed_dealer_hand)

    # Determine move
    move = "Waiting for Cards"
    if confirmed_player_hand and confirmed_dealer_hand:
        move = blackjack_strategy(player_score, list(confirmed_dealer_hand)[0])

    # Get card probabilities
    card_probabilities = get_card_probabilities()

    # Display game info
    cv2.putText(frame, f"Player: {list(confirmed_player_hand)} | Score: {player_score}",
                (20, frame.shape[0] - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Dealer: {list(confirmed_dealer_hand)} | Score: {dealer_score}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"Move: {move}", (20, frame.shape[0] - 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    cv2.putText(frame, f"Cards Used: {used_cards}", (20, frame.shape[0] - 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display card probabilities
    y_offset = frame.shape[0] - 180
    cv2.putText(frame, "Next Card Probabilities:", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    for rank, prob in card_probabilities.items():
        y_offset -= 30
        cv2.putText(frame, f"{rank}: {prob:.2%}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show frame
    cv2.imshow("HackJack", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
