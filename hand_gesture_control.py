import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Disable PyAutoGUI fail-safe
pyautogui.FAILSAFE = False

# Initialize camera
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera!")
    exit()

# Initialize MediaPipe with higher precision
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,  # Slightly reduced for better detection
    min_tracking_confidence=0.7    # Slightly reduced for better tracking
)

# Get screen size
screen_width, screen_height = pyautogui.size()
print(f"Screen size: {screen_width}x{screen_height}")

# Get camera size
camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
camera_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera size: {camera_width}x{camera_height}")

# Enhanced mouse sensitivity and smoothing variables for precision
sensitivity = 1.5  # Reduced for more precise control
prev_x, prev_y = 0, 0
smoothing_factor = 0.95  # Increased smoothing for ultra-smooth movement

# Click control variables - IMPROVED FOR RELIABILITY
click_cooldown = 0.4  # Increased for better reliability
last_click_time = 0
last_right_click_time = 0

# Gesture state variables - IMPROVED FOR RELIABILITY
gesture_hold_time = 0.3  # Increased for better gesture recognition
gesture_start_time = 0
current_gesture = "none"
last_gesture = "none"
gesture_cooldown = 1.0  # Increased for better reliability
last_gesture_time = 0

# Gesture state management - IMPROVED FOR RELIABILITY
awesome_gesture_used = False  # Changed from peace_gesture_used
pinch_gesture_used = False
gesture_reset_time = 0.8  # Increased for better reliability

# Cursor lock during gestures
cursor_locked = False
locked_cursor_x = 0
locked_cursor_y = 0

# Inverted mouse control
invert_x = True
invert_y = False

# Scroll control variables - IMPROVED FOR ONE-TIME SCROLLING
scroll_cooldown = 0.8  # Increased to prevent multiple scrolls
last_scroll_time = 0
prev_index_x = 0
prev_index_y = 0
flick_threshold = 25  # Reduced for easier detection
flick_speed_threshold = 40  # Reduced for easier detection
index_positions = []  # Store recent index finger positions
max_positions = 5  # Number of positions to track
flick_detected = False  # Track if flick was already detected

# Center the cursor initially
pyautogui.moveTo(screen_width//2, screen_height//2)
prev_x, prev_y = screen_width//2, screen_height//2

print("Ultra-Precise Hand Gesture Control Started!")
print("Index finger tip controls cursor with ultra-high precision")
print("Pinch Sign = Left Click, Awesome Sign (Thumbs Up) = Right Click")
print("Index finger flick left = Previous, Index finger flick right = Next")
print("Cursor stays still during gestures")
print("Mouse movement: Left-Right INVERTED, Up-Down NORMAL")
print("Press 'q' to quit")

def calculate_distance(point1, point2):
    """Calculate distance between two points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def is_awesome_sign(hand_landmarks):
    """Check if hand is in awesome sign (thumb up, other fingers down) - IMPROVED"""
    # Get finger tip and pip landmarks
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    
    # Get corresponding pip joints
    thumb_pip = hand_landmarks.landmark[3]
    index_pip = hand_landmarks.landmark[6]
    middle_pip = hand_landmarks.landmark[10]
    ring_pip = hand_landmarks.landmark[14]
    pinky_pip = hand_landmarks.landmark[18]
    
    # Check for thumbs up gesture
    thumb_up = thumb_tip.y < thumb_pip.y - 0.01  # Thumb pointing up
    index_down = index_tip.y > index_pip.y + 0.01  # Index finger down
    middle_down = middle_tip.y > middle_pip.y + 0.01  # Middle finger down
    ring_down = ring_tip.y > ring_pip.y + 0.01  # Ring finger down
    pinky_down = pinky_tip.y > pinky_pip.y + 0.01  # Pinky finger down
    
    # Awesome sign detection (thumbs up)
    return thumb_up and index_down and middle_down and ring_down and pinky_down

def is_pinch_sign(hand_landmarks):
    """Check if hand is in pinch sign (thumb and index finger touching) - MORE RELIABLE"""
    # Get thumb tip and index finger tip
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    
    # Calculate distance between thumb and index finger
    distance = calculate_distance(
        (thumb_tip.x * camera_width, thumb_tip.y * camera_height),
        (index_tip.x * camera_width, index_tip.y * camera_height)
    )
    
    # Check if other fingers are down (more lenient)
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    
    middle_pip = hand_landmarks.landmark[10]
    ring_pip = hand_landmarks.landmark[14]
    pinky_pip = hand_landmarks.landmark[18]
    
    # More lenient finger down detection
    middle_down = middle_tip.y > middle_pip.y + 0.01  # Reduced margin
    ring_down = ring_tip.y > ring_pip.y + 0.01  # Reduced margin
    pinky_down = pinky_tip.y > pinky_pip.y + 0.01  # Reduced margin
    
    # More lenient pinch threshold for better reliability
    pinch_threshold = 60  # pixels - increased for better detection
    
    # Return True if fingers are close enough and other fingers are down
    return distance < pinch_threshold and middle_down and ring_down and pinky_down

def map_hand_to_screen(hand_x, hand_y):
    """Map hand coordinates to screen coordinates with ultra-precision"""
    # Minimal padding for maximum screen coverage
    padding = 0.02  # 2% padding from edges (more precise control)
    
    # Map hand position to screen with padding
    screen_x = hand_x * (screen_width * (1 - 2 * padding)) + (screen_width * padding)
    screen_y = hand_y * (screen_height * (1 - 2 * padding)) + (screen_height * padding)
    
    return int(screen_x), int(screen_y)

def invert_coordinates(x, y):
    """Invert the coordinates for mouse control"""
    # Invert X coordinate (left-right)
    if invert_x:
        x = 1.0 - x
    
    return x, y

def clamp_coordinates(x, y):
    """Clamp coordinates to screen bounds"""
    x = max(0, min(x, screen_width))
    y = max(0, min(y, screen_height))
    return x, y

def apply_ultra_precision_smoothing(x, y, prev_x, prev_y):
    """Apply ultra-precision smoothing for smooth movement"""
    # Multi-stage smoothing for ultra-smooth movement
    alpha1 = 0.90  # First stage smoothing
    alpha2 = 0.98  # Second stage smoothing
    alpha3 = 0.99  # Third stage smoothing
    
    # First stage smoothing
    smooth_x1 = alpha1 * x + (1 - alpha1) * prev_x
    smooth_y1 = alpha1 * y + (1 - alpha1) * prev_y
    
    # Second stage smoothing
    smooth_x2 = alpha2 * smooth_x1 + (1 - alpha2) * prev_x
    smooth_y2 = alpha2 * smooth_y1 + (1 - alpha2) * prev_y
    
    # Third stage smoothing
    smooth_x3 = alpha3 * smooth_x2 + (1 - alpha3) * prev_x
    smooth_y3 = alpha3 * smooth_y2 + (1 - alpha3) * prev_y
    
    return int(smooth_x3), int(smooth_y3)

def apply_dead_zone(x, y, center_x, center_y):
    """Apply dead zone to prevent jittery movement"""
    dead_zone = 2  # pixels
    
    dx = x - center_x
    dy = y - center_y
    
    if abs(dx) < dead_zone:
        dx = 0
    if abs(dy) < dead_zone:
        dy = 0
    
    return center_x + dx, center_y + dy

def detect_single_flick(current_x, current_y, current_time, last_flick_time):
    """Detect single flick for one-time scrolling"""
    global index_positions, flick_detected
    
    # Add current position to tracking list
    index_positions.append((current_x, current_y, current_time))
    
    # Keep only recent positions
    if len(index_positions) > max_positions:
        index_positions.pop(0)
    
    # Need at least 3 positions to detect flick
    if len(index_positions) < 3:
        return None, last_flick_time
    
    # Check cooldown
    if current_time - last_flick_time < scroll_cooldown:
        return None, last_flick_time
    
    # If flick was already detected, wait for reset
    if flick_detected:
        return None, last_flick_time
    
    # Calculate total horizontal movement
    first_pos = index_positions[0]
    last_pos = index_positions[-1]
    
    total_dx = last_pos[0] - first_pos[0]
    total_dy = last_pos[1] - first_pos[1]
    
    # Calculate total distance
    total_distance = calculate_distance((first_pos[0], first_pos[1]), (last_pos[0], last_pos[1]))
    
    # Calculate time span
    time_span = last_pos[2] - first_pos[2]
    
    # Calculate speed (pixels per second)
    if time_span > 0:
        speed = total_distance / time_span
    else:
        speed = 0
    
    # Check if it's a horizontal flick
    if (abs(total_dx) > abs(total_dy) and  # More horizontal than vertical
        abs(total_dx) > flick_threshold and  # Enough horizontal movement
        speed > flick_speed_threshold):  # Fast enough
        
        # Mark flick as detected
        flick_detected = True
        
        if total_dx > 0:  # Flick right
            return "right", current_time
        else:  # Flick left
            return "left", current_time
    
    return None, last_flick_time

def reset_flick_detection():
    """Reset flick detection after cooldown"""
    global flick_detected, index_positions
    flick_detected = False
    index_positions = []  # Clear position history

while True:
    # Read frame
    ret, frame = cap.read()
    
    # Check if frame was successfully captured
    if not ret or frame is None:
        print("Error: Could not read frame!")
        break
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = hands.process(rgb_frame)
    
    # If hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get index finger tip (landmark 8) for cursor control
            index_tip = hand_landmarks.landmark[8]
            
            # Get thumb tip (landmark 4) for pinch detection
            thumb_tip = hand_landmarks.landmark[4]
            
            # Invert coordinates (only X axis)
            control_x, control_y = invert_coordinates(index_tip.x, index_tip.y)
            
            # Convert to pixel coordinates for display
            x = int(control_x * camera_width)
            y = int(control_y * camera_height)
            
            # Convert thumb coordinates for pinch detection
            thumb_x = int(thumb_tip.x * camera_width)
            thumb_y = int(thumb_tip.y * camera_height)
            
            # Map hand coordinates to screen coordinates with ultra-precision
            screen_x, screen_y = map_hand_to_screen(control_x, control_y)
            
            # Apply enhanced sensitivity
            center_x, center_y = screen_width // 2, screen_height // 2
            screen_x = center_x + (screen_x - center_x) * sensitivity
            screen_y = center_y + (screen_y - center_y) * sensitivity
            
            # Apply dead zone to prevent jitter
            screen_x, screen_y = apply_dead_zone(screen_x, screen_y, center_x, center_y)
            
            # Clamp coordinates to screen bounds
            screen_x, screen_y = clamp_coordinates(screen_x, screen_y)
            
            # Apply ultra-precision smoothing
            screen_x, screen_y = apply_ultra_precision_smoothing(screen_x, screen_y, prev_x, prev_y)
            
            # Clamp smoothed coordinates
            screen_x, screen_y = clamp_coordinates(screen_x, screen_y)
            
            # Check for gestures first
            current_time = time.time()
            
            # Check for pinch sign (left click) - IMPROVED RELIABILITY
            pinch_detected = is_pinch_sign(hand_landmarks)
            
            # Check for awesome sign (right click) - IMPROVED RELIABILITY
            awesome_detected = is_awesome_sign(hand_landmarks)
            
            # Detect single flick gesture for one-time scrolling
            flick_direction, last_scroll_time = detect_single_flick(x, y, current_time, last_scroll_time)
            
            # Handle single scroll gesture
            if flick_direction:
                if flick_direction == "right":
                    pyautogui.scroll(-1)  #  scroll up (previous)
                    print("Scroll Next")
                elif flick_direction == "left":
                    pyautogui.scroll(1)   # Single scroll down (next)
                    print("Scroll previous")
                
                # Reset flick detection after a delay
                time.sleep(0.5)  # Small delay
                reset_flick_detection()
            
            # Determine if we should lock cursor
            if awesome_detected or pinch_detected:
                if not cursor_locked:
                    cursor_locked = True
                    locked_cursor_x = prev_x
                    locked_cursor_y = prev_y
            else:
                cursor_locked = False
            
            # Move mouse cursor (only if not locked)
            if not cursor_locked:
                try:
                    pyautogui.moveTo(screen_x, screen_y)
                    prev_x, prev_y = screen_x, screen_y
                except Exception as e:
                    print(f"Mouse movement error: {e}")
                    continue
            else:
                # Keep cursor at locked position
                try:
                    pyautogui.moveTo(locked_cursor_x, locked_cursor_y)
                except Exception as e:
                    print(f"Mouse movement error: {e}")
                    continue
            
            # Handle pinch gesture (left click) - IMPROVED RELIABILITY
            if pinch_detected:
                if not pinch_gesture_used and (current_time - last_click_time) > click_cooldown:
                    try:
                        pyautogui.click()
                        last_click_time = current_time
                        pinch_gesture_used = True
                        print("Left Click! (Pinch sign)")
                    except Exception as e:
                        print(f"Click error: {e}")
            
            # Handle awesome sign gesture (right click) - IMPROVED RELIABILITY
            elif awesome_detected:
                if not awesome_gesture_used and (current_time - last_right_click_time) > click_cooldown:
                    try:
                        pyautogui.rightClick()
                        last_right_click_time = current_time
                        awesome_gesture_used = True
                        print("Right Click! (Awesome sign)")
                    except Exception as e:
                        print(f"Right click error: {e}")
            
            # Reset gesture states when no gesture is detected - IMPROVED RELIABILITY
            if not pinch_detected and not awesome_detected:
                if current_time - last_click_time > gesture_reset_time:
                    pinch_gesture_used = False
                if current_time - last_right_click_time > gesture_reset_time:
                    awesome_gesture_used = False
            
            # Draw circle on index finger tip
            if cursor_locked:
                circle_color = (0, 0, 255)  # Red when cursor is locked
            else:
                circle_color = (0, 255, 0)  # Green for normal mode
            circle_size = 6  # Smaller circle for precision
            cv2.circle(frame, (x, y), circle_size, circle_color, -1)
            
            # Draw circle on thumb tip
            cv2.circle(frame, (thumb_x, thumb_y), 5, (255, 255, 0), -1)  # Yellow for thumb
            
            # Draw line between thumb and index finger for pinch visualization
            pinch_distance = calculate_distance(
                (thumb_x, thumb_y),
                (x, y)
            )
            if pinch_distance < 80:  # Show line when fingers are close
                line_color = (0, 255, 255) if pinch_distance < 60 else (255, 0, 255)  # Green if pinching, purple if close
                cv2.line(frame, (thumb_x, thumb_y), (x, y), line_color, 2)
            
            # Draw hand landmarks
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Display frame
    cv2.imshow('Ultra-Precise Hand Gesture Control', frame)
    
    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Program stopped!")
