import cv2
import mediapipe as mp
import numpy as np
import time
import math
import winsound
import threading
import pyautogui

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0. 

# ==================== SETUP ====================
mp_hands = mp.solutions.hands
# Lower detection confidence for better tracking, higher tracking confidence for stability
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,  # Lower for better detection
    min_tracking_confidence=0.7    # Moderate tracking confidence
)    
mp_draw = mp.solutions

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cv2.namedWindow("Gesture Drawing Pro", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Gesture Drawing Pro", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Canvas and drawing states
canvas = None
prev_x, prev_y = None, None
eraser_prev_x, eraser_prev_y = None, None
eraser_size = 30

# Move/grab state
# grabbed_region = None
# grab_start_pos = None
# grab_hand_id = None
# grabbed_bbox = None
# move_mode = False

# AI state
detected_objects = []
ai_suggestions = []
search_query = ""
typing_mode = False

# YouTube mode state
youtube_mode = False
youtube_activated_time = 0
youtube_activation_threshold = 1.0  # seconds to hold gesture to activate
youtube_opened = False
youtube_open_time = 0
last_index_extended = {}  # Track per hand

# Gesture tracking state for smoothing
gesture_history = {}  # Track gesture stability
GESTURE_CONFIRMATION_FRAMES = 3  # Require gesture for N frames to confirm

# Palm history for swipe detection
palm_history = {}
last_swipe_time = 0
swipe_threshold = 120  # pixels
swipe_time_window = 0.5  # seconds

# Hand tracking cache for normalization
hand_size_cache = {}  # Cache hand sizes for normalization
hand_motion_cache = {}  # Cache for motion filtering
KALMAN_ALPHA = 0.5  # Kalman filter alpha (higher = more responsive, lower = smoother)

# Colors
COLOR_BG = (20, 20, 30)
COLOR_ACCENT = (0, 200, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_SUCCESS = (0, 255, 100)
COLOR_WARNING = (0, 165, 255)
COLOR_DRAW = (50, 100, 255)

# Audio beep functions
def beep(frequency=1000, duration=200):
    """Play a beep sound in a separate thread to avoid blocking"""
    def _beep():
        try:
            winsound.Beep(frequency, duration)
        except:
            pass
    thread = threading.Thread(target=_beep, daemon=True)
    thread.start()

def beep_activate():
    """Beep for mode activation (higher pitch)"""
    beep(1500, 150)

def beep_gesture():
    """Beep for gesture recognition (normal pitch)"""
    beep(1000, 150)

def youtube_press(key):
    try:
        pyautogui.press(key)
    except Exception:
        pass

def youtube_hotkey(*keys):
    try:
        pyautogui.hotkey(*keys)
    except Exception:
        pass

def open_youtube():
    try:
        import os
        import webbrowser
        if os.name == 'nt':
            os.startfile("https://www.youtube.com")
        else:
            webbrowser.open("https://www.youtube.com", new=2)
        print("Opening YouTube")
        return True
    except Exception as e:
        print(f"Failed to open YouTube: {e}")
        try:
            import webbrowser
            webbrowser.open("https://www.youtube.com", new=2)
            print("Opening YouTube via webbrowser")
            return True
        except Exception as e2:
            print(f"Failed via webbrowser too: {e2}")
            return False

def show_finger_status(frame, lm, palm_x, palm_y, youtube_mode_active):
    """Display finger status for debugging gesture recognition"""
    thumb_up = lm.landmark[4].y < lm.landmark[3].y
    index_up = lm.landmark[8].y < lm.landmark[6].y
    middle_up = lm.landmark[12].y < lm.landmark[10].y
    ring_up = lm.landmark[16].y < lm.landmark[14].y
    pinky_up = lm.landmark[20].y < lm.landmark[18].y
    
    status_text = f"Fingers - Thumb:{'U' if thumb_up else 'D'} Index:{'U' if index_up else 'D'} Mid:{'U' if middle_up else 'D'} Ring:{'U' if ring_up else 'D'} Pinky:{'U' if pinky_up else 'D'}"
    color = COLOR_SUCCESS if youtube_mode_active else COLOR_TEXT
    cv2.putText(frame, status_text, (palm_x - 200, palm_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def get_hand_size(lm):
    """Calculate hand size for normalization"""
    # Distance from wrist to middle finger tip
    wrist = lm.landmark[0]
    middle_tip = lm.landmark[12]
    hand_size = math.sqrt((middle_tip.x - wrist.x)**2 + (middle_tip.y - wrist.y)**2)
    return max(hand_size, 0.1)  # Avoid division by zero

def get_normalized_threshold(hand_size, base_threshold=0.05):
    """Get adaptive threshold based on hand size"""
    # Standard hand size is approximately 0.35
    standard_size = 0.35
    return base_threshold * (hand_size / standard_size)

def is_finger_up(tip_idx, pip_idx, lm, hand_size):
    """Check if finger is up with adaptive threshold"""
    threshold = get_normalized_threshold(hand_size, 0.04)
    return lm.landmark[tip_idx].y < (lm.landmark[pip_idx].y - threshold)

def is_finger_down(tip_idx, pip_idx, lm, hand_size):
    """Check if finger is down with adaptive threshold"""
    threshold = get_normalized_threshold(hand_size, 0.04)
    return lm.landmark[tip_idx].y > (lm.landmark[pip_idx].y - threshold)

# Gesture helpers

def is_index_up(lm):
    return lm.landmark[8].y < lm.landmark[6].y

def is_palm_open(lm):
    hand_size = get_hand_size(lm)
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    threshold = get_normalized_threshold(hand_size, 0.06)  # Slightly more lenient
    open_fingers = all(lm.landmark[tips[i]].y < (lm.landmark[pips[i]].y - threshold) for i in range(4))
    # Removed thumb requirement for palm open gesture
    return open_fingers

def is_two_fingers_up(lm):
    """Index and middle fingers up, others down"""
    hand_size = get_hand_size(lm)
    idx_up = is_finger_up(8, 6, lm, hand_size)
    mid_up = is_finger_up(12, 10, lm, hand_size)
    ring_down = is_finger_down(16, 14, lm, hand_size)
    pinky_down = is_finger_down(20, 18, lm, hand_size)
    return idx_up and mid_up and ring_down and pinky_down

def is_fist_closed(lm):
    """All fingers closed (fist gesture for play/pause)"""
    hand_size = get_hand_size(lm)
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    threshold = get_normalized_threshold(hand_size, 0.05)
    fingers_closed = all(lm.landmark[tips[i]].y > (lm.landmark[pips[i]].y - threshold) for i in range(4))
    thumb_closed = lm.landmark[4].y > lm.landmark[3].y  # Thumb tip below IP joint
    return fingers_closed and thumb_closed

def is_thumb_and_pinky_up(lm):
    """Thumb and pinky fingers extended, others closed - activates YouTube mode"""
    hand_size = get_hand_size(lm)
    # Check if thumb is extended upward
    thumb_up = is_finger_up(4, 2, lm, hand_size)
    
    # Check if pinky is extended upward
    pinky_up = is_finger_up(20, 18, lm, hand_size)
    
    # Check other fingers are closed
    index_closed = is_finger_down(8, 6, lm, hand_size)
    middle_closed = is_finger_down(12, 10, lm, hand_size)
    ring_closed = is_finger_down(16, 14, lm, hand_size)
    
    return thumb_up and pinky_up and index_closed and middle_closed and ring_closed

def is_index_middle_up(lm):
    """Index and middle fingers up - Next video"""
    hand_size = get_hand_size(lm)
    index_up = is_finger_up(8, 6, lm, hand_size)
    middle_up = is_finger_up(12, 10, lm, hand_size)
    ring_down = is_finger_down(16, 14, lm, hand_size)
    pinky_down = is_finger_down(20, 18, lm, hand_size)
    thumb_down = is_finger_down(4, 2, lm, hand_size)
    return index_up and middle_up and ring_down and pinky_down and thumb_down

def is_thumb_index_middle_up(lm):
    """Thumb, index, and middle fingers up - Previous video"""
    hand_size = get_hand_size(lm)
    thumb_up = lm.landmark[4].y < lm.landmark[3].y - 0.03  # More lenient for thumb
    index_up = is_finger_up(8, 6, lm, hand_size)
    middle_up = is_finger_up(12, 10, lm, hand_size)
    ring_down = is_finger_down(16, 14, lm, hand_size)
    pinky_down = is_finger_down(20, 18, lm, hand_size)
    return thumb_up and index_up and middle_up and ring_down and pinky_down

def is_four_fingers_up(lm):
    """Four fingers up (index, middle, ring, pinky) - Scroll YouTube home"""
    hand_size = get_hand_size(lm)
    index_up = is_finger_up(8, 6, lm, hand_size)
    middle_up = is_finger_up(12, 10, lm, hand_size)
    ring_up = is_finger_up(16, 14, lm, hand_size)
    pinky_up = is_finger_up(20, 18, lm, hand_size)
    thumb_down = is_finger_down(4, 2, lm, hand_size)
    return index_up and middle_up and ring_up and pinky_up and thumb_down

def is_thumbs_up(lm):
    """Thumbs up gesture - increase volume YouTube"""
    hand_size = get_hand_size(lm)
    thumb_tip = lm.landmark[4]
    thumb_pip = lm.landmark[3]
    palm_center = (lm.landmark[0].y + lm.landmark[9].y) / 2
    
    # Thumb pointing up relative to palm
    thumb_up = thumb_tip.y < thumb_pip.y
    threshold = get_normalized_threshold(hand_size, 0.15)
    thumb_extended = thumb_tip.y < (palm_center - threshold)
    
    # Other fingers closed
    fingers_closed = all(is_finger_down(tips, pips, lm, hand_size) 
                         for tips, pips in [(8, 6), (12, 10), (16, 14), (20, 18)])
    
    return thumb_up and thumb_extended and fingers_closed

def is_pinky_up(lm):
    """Pinky finger up - decrease volume YouTube"""
    hand_size = get_hand_size(lm)
    pinky_tip = lm.landmark[20]
    pinky_pip = lm.landmark[18]
    palm_center = (lm.landmark[0].y + lm.landmark[9].y) / 2
    
    # Pinky pointing up
    pinky_extended_up = pinky_tip.y < pinky_pip.y
    threshold = get_normalized_threshold(hand_size, 0.1)
    pinky_high = pinky_tip.y < (palm_center - threshold)
    
    # Other fingers closed (including thumb)
    other_closed = all(is_finger_down(tips, pips, lm, hand_size) 
                       for tips, pips in [(4, 3), (8, 6), (12, 10), (16, 14)])
    
    return pinky_extended_up and pinky_high and other_closed

def get_hand_bounding_box(lm):
    """Get bounding box of hand for validation"""
    x_coords = [lm.landmark[i].x for i in range(21)]
    y_coords = [lm.landmark[i].y for i in range(21)]
    return min(x_coords), max(x_coords), min(y_coords), max(y_coords)

def is_valid_hand(lm):
    """Validate hand landmarks to filter out false detections"""
    x_min, x_max, y_min, y_max = get_hand_bounding_box(lm)
    width = x_max - x_min
    height = y_max - y_min
    area = width * height
    
    # Hand too small or partially visible
    if width < 0.04 or height < 0.04 or area < 0.0012:
        return False
    
    aspect_ratio = width / height if height > 0 else 1
    # Allow more rotated and natural hand shapes
    if aspect_ratio < 0.18 or aspect_ratio > 5.5:
        return False
    
    return True

def get_palm_center(lm):
    wrist = lm.landmark[0]
    middle = lm.landmark[9]
    return (wrist.x + middle.x) / 2, (wrist.y + middle.y) / 2

# AI utility functions

def detect_objects_ai(frame):
    """Simple color‑based object detection"""
    if frame is None:
        return []
    objs = []
    small = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    colors = {
        "Red": (np.array([0, 100, 100]), np.array([10, 255, 255])),
        "Green": (np.array([40, 100, 100]), np.array([80, 255, 255])),
        "Blue": (np.array([100, 100, 100]), np.array([130, 255, 255]))
    }
    for name,(low,high) in colors.items():
        mask = cv2.inRange(hsv, low, high)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) > 200:
                x,y,w,h = cv2.boundingRect(c)
                objs.append((name, x*2, y*2, w*2, h*2))
    return objs

def get_drawing_suggestions():
    options = [
        "Draw a circle",
        "Sketch a star",
        "Try a smiling face",
        "Add a tree",
        "Build a house",
        "Paint a sun",
    ]
    return options

def draw_house(img, x, y, size=100, color=(0,255,255)):
    """Draw a simple house at (x,y) with given size."""
    # base
    cv2.rectangle(img, (x, y), (x + size, y + size), color, 2)
    # roof
    pts = np.array([[x, y], [x + size//2, y - size//2], [x + size, y]], np.int32)
    cv2.fillPoly(img, [pts], (color[0], color[1], color[2]-50))

def draw_circle(img, x, y, radius=50, color=(0,255,255)):
    """Draw a circle at (x,y) with given radius."""
    cv2.circle(img, (x, y), radius, color, 2)

def draw_square(img, x, y, size=80, color=(255,0,255)):
    """Draw a square at (x,y) with given size."""
    cv2.rectangle(img, (x-size//2, y-size//2), (x+size//2, y+size//2), color, 2)

def draw_rectangle(img, x, y, width=120, height=60, color=(255,255,0)):
    """Draw a rectangle at (x,y)."""
    cv2.rectangle(img, (x-width//2, y-height//2), (x+width//2, y+height//2), color, 2)

def draw_triangle(img, x, y, size=60, color=(0,255,0)):
    """Draw a triangle at (x,y) with given size."""
    pts = np.array([[x, y-size], [x-size, y+size], [x+size, y+size]], np.int32)
    cv2.polylines(img, [pts], True, color, 2)

def draw_pentagon(img, x, y, size=50, color=(200,100,255)):
    """Draw a pentagon at (x,y)."""
    pts = []
    for i in range(5):
        angle = 2 * np.pi * i / 5 - np.pi / 2
        px = int(x + size * np.cos(angle))
        py = int(y + size * np.sin(angle))
        pts.append([px, py])
    pts = np.array(pts, np.int32)
    cv2.polylines(img, [pts], True, color, 2)

def draw_star(img, x, y, size=60, color=(100,200,255)):
    """Draw a star at (x,y) with given size."""
    pts = []
    for i in range(10):
        angle = 2 * np.pi * i / 10 - np.pi / 2
        r = size if i % 2 == 0 else size // 2
        px = int(x + r * np.cos(angle))
        py = int(y + r * np.sin(angle))
        pts.append([px, py])
    pts = np.array(pts, np.int32)
    cv2.polylines(img, [pts], True, color, 2)

def draw_ellipse(img, x, y, width=100, height=60, color=(150,150,255)):
    """Draw an ellipse at (x,y)."""
    cv2.ellipse(img, (x, y), (width//2, height//2), 0, 0, 360, color, 2)

def draw_heart(img, x, y, size=40, color=(100,255,255)):
    """Draw a heart at (x,y)."""
    # Top left bulge
    cv2.circle(img, (x - size//3, y - size//3), size//3, color, 2)
    # Top right bulge
    cv2.circle(img, (x + size//3, y - size//3), size//3, color, 2)
    # Bottom point
    pts = np.array([[x - size, y], [x, y + size], [x + size, y]], np.int32)
    cv2.polylines(img, [pts], True, color, 2)

def draw_diamond(img, x, y, size=60, color=(50,200,150)):
    """Draw a diamond at (x,y)."""
    pts = np.array([[x, y-size], [x+size, y], [x, y+size], [x-size, y]], np.int32)
    cv2.polylines(img, [pts], True, color, 2)

SHAPE_FUNCTIONS = {
    "circle": lambda img, x, y: draw_circle(img, x, y, radius=50),
    "square": lambda img, x, y: draw_square(img, x, y, size=80),
    "rectangle": lambda img, x, y: draw_rectangle(img, x, y, width=120, height=60),
    "triangle": lambda img, x, y: draw_triangle(img, x, y, size=60),
    "pentagon": lambda img, x, y: draw_pentagon(img, x, y, size=50),
    "star": lambda img, x, y: draw_star(img, x, y, size=60),
    "ellipse": lambda img, x, y: draw_ellipse(img, x, y, width=100, height=60),
    "heart": lambda img, x, y: draw_heart(img, x, y, size=40),
    "diamond": lambda img, x, y: draw_diamond(img, x, y, size=60),
}

# main loop with smoothing and multi-hand support
frame_count=0; fps_time=time.time(); fps=0
# smoothing & per-hand state
prev_positions = {}
prev_draw = {}
prev_cursor = {}
cursor_alpha = 0.25  # cursor smoothing factor
alpha = 0.45  # smoothing factor (lower = smoother, more stable tracking)
while True:
    ok, frame = cap.read()
    if not ok: break
    frame_count += 1
    if time.time() - fps_time > 1:
        fps = frame_count; frame_count = 0; fps_time = time.time()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8); canvas[:] = COLOR_BG
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    fx = fy = None
    if results.multi_hand_landmarks:
        best_index = None
        best_lm = None
        best_area = 0.0
        for i, lm in enumerate(results.multi_hand_landmarks):
            x_min, x_max, y_min, y_max = get_hand_bounding_box(lm)
            area = (x_max - x_min) * (y_max - y_min)
            if area > best_area and is_valid_hand(lm):
                best_area = area
                best_index = i
                best_lm = lm
        if best_lm is not None:
            i = best_index
            lm = best_lm
            hand_label = "Unknown"
            if results.multi_handedness and i < len(results.multi_handedness):
                hand_label = results.multi_handedness[i].classification[0].label
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            raw_fx = int(lm.landmark[8].x * w); raw_fy = int(lm.landmark[8].y * h)
            if i not in prev_positions:
                smooth_fx, smooth_fy = raw_fx, raw_fy
            else:
                old_x, old_y = prev_positions[i]
                smooth_fx = int(alpha * raw_fx + (1 - alpha) * old_x)
                smooth_fy = int(alpha * raw_fy + (1 - alpha) * old_y)
            prev_positions[i] = (smooth_fx, smooth_fy)
            fx, fy = smooth_fx, smooth_fy
            palm_x, palm_y = get_palm_center(lm)
            palm_x = int(palm_x * w); palm_y = int(palm_y * h)
            cv2.putText(frame, f"{hand_label}", (smooth_fx + 10, smooth_fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            
            # Track palm history for swipe detection
            if i not in palm_history:
                palm_history[i] = []
            palm_history[i].append((palm_x, palm_y, time.time()))
            if len(palm_history[i]) > 20:
                palm_history[i].pop(0)
            
            # Show finger status for debugging
            show_finger_status(frame, lm, palm_x, palm_y, youtube_mode)
            # erasing
            if is_palm_open(lm) and not youtube_mode:
                if eraser_prev_x is not None:
                    cv2.line(canvas, (eraser_prev_x, eraser_prev_y), (palm_x, palm_y), COLOR_BG, eraser_size)
                eraser_prev_x, eraser_prev_y = palm_x, palm_y
                cv2.circle(frame, (palm_x, palm_y), eraser_size // 2, COLOR_WARNING, 2)
            else:
                eraser_prev_x = eraser_prev_y = None

            if youtube_mode:
                palm_open = is_palm_open(lm)
                print(f"Palm open check: {palm_open}")  # Debug print
                if palm_open:
                    print("Palm open detected in YouTube mode")
                    if youtube_open_time == 0:
                        youtube_open_time = time.time()
                        print("Starting YouTube open timer")
                    elif time.time() - youtube_open_time > youtube_activation_threshold and not youtube_opened:
                        if open_youtube():
                            youtube_opened = True
                            beep_activate()
                            cv2.putText(frame, "OPENING YOUTUBE...", (palm_x - 100, palm_y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_SUCCESS, 2)
                        else:
                            cv2.putText(frame, "FAILED TO OPEN YOUTUBE", (palm_x - 160, palm_y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_WARNING, 2)
                else:
                    youtube_open_time = 0
            
            # YouTube mode activation: thumb and pinky gesture
            if is_thumb_and_pinky_up(lm):
                if youtube_activated_time == 0:
                    youtube_activated_time = time.time()
                elif time.time() - youtube_activated_time > youtube_activation_threshold:
                    youtube_mode = not youtube_mode
                    youtube_opened = False
                    youtube_activated_time = 0
                    beep_activate()  # Play activation beep
                    cv2.putText(frame, f"YOUTUBE MODE: {'ON' if youtube_mode else 'OFF'}", (palm_x - 100, palm_y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_SUCCESS, 2)
                cv2.circle(frame, (palm_x, palm_y), 30, (0, 0, 255), 2)
            else:
                youtube_activated_time = 0
            
            # YouTube controls in YouTube mode
            if youtube_mode:
                # Next video: Index + Middle
                if is_index_middle_up(lm):
                    if i not in gesture_history:
                        gesture_history[i] = {"next": 0, "next_beep": False}
                    gesture_history[i]["next"] = gesture_history[i].get("next", 0) + 1
                    if gesture_history[i]["next"] >= GESTURE_CONFIRMATION_FRAMES:
                        if not gesture_history[i].get("next_beep", False):
                            beep_gesture()  # Play gesture beep
                            youtube_hotkey('shift', 'n')
                            gesture_history[i]["next_beep"] = True
                        cv2.putText(frame, "NEXT VIDEO", (palm_x - 60, palm_y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_SUCCESS, 2)
                else:
                    if i in gesture_history:
                        gesture_history[i]["next"] = 0
                        gesture_history[i]["next_beep"] = False
                
                # Previous video: Thumb + Index + Middle
                if is_thumb_index_middle_up(lm):
                    if i not in gesture_history:
                        gesture_history[i] = {"prev": 0, "prev_beep": False}
                    gesture_history[i]["prev"] = gesture_history[i].get("prev", 0) + 1
                    if gesture_history[i]["prev"] >= GESTURE_CONFIRMATION_FRAMES:
                        if not gesture_history[i].get("prev_beep", False):
                            beep_gesture()  # Play gesture beep
                            youtube_hotkey('shift', 'p')
                            gesture_history[i]["prev_beep"] = True
                        cv2.putText(frame, "PREVIOUS VIDEO", (palm_x - 80, palm_y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WARNING, 2)
                else:
                    if i in gesture_history:
                        gesture_history[i]["prev"] = 0
                        gesture_history[i]["prev_beep"] = False
                
                # Play/Pause: Fist closed
                if is_fist_closed(lm):
                    if i not in gesture_history:
                        gesture_history[i] = {"play_pause": 0, "play_pause_beep": False}
                    gesture_history[i]["play_pause"] = gesture_history[i].get("play_pause", 0) + 1
                    if gesture_history[i]["play_pause"] >= GESTURE_CONFIRMATION_FRAMES:
                        if not gesture_history[i].get("play_pause_beep", False):
                            beep_gesture()  # Play gesture beep
                            youtube_press('space')
                            gesture_history[i]["play_pause_beep"] = True
                        cv2.putText(frame, "PLAY/PAUSE", (palm_x - 70, palm_y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_ACCENT, 2)
                else:
                    if i in gesture_history:
                        gesture_history[i]["play_pause"] = 0
                        gesture_history[i]["play_pause_beep"] = False
                
                # Scroll YouTube home: 4 fingers
                if is_four_fingers_up(lm):
                    if i not in gesture_history:
                        gesture_history[i] = {"scroll": 0, "scroll_beep": False}
                    gesture_history[i]["scroll"] = gesture_history[i].get("scroll", 0) + 1
                    if gesture_history[i]["scroll"] >= GESTURE_CONFIRMATION_FRAMES:
                        if not gesture_history[i].get("scroll_beep", False):
                            beep_gesture()  # Play gesture beep
                            youtube_press('pagedown')
                            gesture_history[i]["scroll_beep"] = True
                        cv2.putText(frame, "SCROLL HOME", (palm_x - 70, palm_y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_ACCENT, 2)
                else:
                    if i in gesture_history:
                        gesture_history[i]["scroll"] = 0
                        gesture_history[i]["scroll_beep"] = False
                
                # Double-click with index finger curl
                index_extended = lm.landmark[8].y < lm.landmark[6].y
                if i not in last_index_extended:
                    last_index_extended[i] = True
                if index_extended:
                    last_index_extended[i] = True
                else:
                    if last_index_extended[i]:
                        pyautogui.doubleClick()
                        beep_gesture()
                        cv2.putText(frame, "DOUBLE CLICK", (palm_x - 80, palm_y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WARNING, 2)
                        last_index_extended[i] = False
                
                # Move mouse cursor with index finger and smooth the motion only when index is extended
                if fx is not None and fy is not None and index_extended:
                    if i not in prev_cursor:
                        cursor_fx, cursor_fy = fx, fy
                    else:
                        old_cx, old_cy = prev_cursor[i]
                        cursor_fx = int(cursor_alpha * fx + (1 - cursor_alpha) * old_cx)
                        cursor_fy = int(cursor_alpha * fy + (1 - cursor_alpha) * old_cy)
                    prev_cursor[i] = (cursor_fx, cursor_fy)
                    pyautogui.moveTo(cursor_fx, cursor_fy)
                else:
                    if i in prev_cursor:
                        del prev_cursor[i]
            
            # drawing per hand with jitter threshold (trigger: two fingers up) - only in drawing mode
            if not youtube_mode and is_two_fingers_up(lm) and not is_palm_open(lm):
                prev = prev_draw.get(i)
                if prev is not None:
                    dx = fx - prev[0]; dy = fy - prev[1]
                    if math.hypot(dx, dy) > 3:
                        cv2.line(canvas, prev, (fx, fy), COLOR_DRAW, 6)
                prev_draw[i] = (fx, fy)
            else:
                prev_draw[i] = None
           
            # grab and move logic (fist gesture) - DISABLED
            # if is_fist_closed(lm):
            #     if grab_hand_id is None or grab_hand_id == i:
            #         grab_hand_id = i
            #         move_mode = True
            #         if grabbed_region is None:
            #             # Start grab: capture region around palm
            #             grab_start_pos = (palm_x, palm_y)
            #             grab_size = 120
            #             x1 = max(0, palm_x - grab_size)
            #             y1 = max(0, palm_y - grab_size)
            #             x2 = min(w, palm_x + grab_size)
            #             y2 = min(h, palm_y + grab_size)
            #             grabbed_region = canvas[y1:y2, x1:x2].copy()
            #             grabbed_bbox = (x1, y1, x2, y2)
            #             # Erase grabbed region from canvas
            #             canvas[y1:y2, x1:x2] = COLOR_BG
            #             cv2.putText(frame, "GRABBED!", (palm_x - 30, palm_y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            #         else:
            #             # Update drag position
            #             grab_size = 120
            #             x1_new = max(0, palm_x - grab_size)
            #             y1_new = max(0, palm_y - grab_size)
            #             x2_new = min(w, palm_x + grab_size)
            #             y2_new = min(h, palm_y + grab_size)
            #             # Draw grabbed region at new position
            #             if grabbed_bbox is not None:
            #                 region_h, region_w = grabbed_region.shape[:2]
            #                 try:
            #                     canvas[y1_new:y1_new+region_h, x1_new:x1_new+region_w] = grabbed_region
            #                 except:
            #                     pass  # ignore resize mismatches
            #             cv2.putText(frame, "DRAGGING...", (palm_x - 40, palm_y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)
            # else:
            #     # Release grab
            #     if grab_hand_id == i and move_mode:
            #         if grabbed_region is not None:
            #             # Place at final position
            #             grab_size = 120
            #             x1_final = max(0, palm_x - grab_size)
            #             y1_final = max(0, palm_y - grab_size)
            #             region_h, region_w = grabbed_region.shape[:2]
            #             try:
            #                 canvas[y1_final:y1_final+region_h, x1_final:x1_final+region_w] = grabbed_region
            #             except:
            #                 pass
            #         grabbed_region = None
            #         grab_start_pos = None
            #         grabbed_bbox = None
            #         grab_hand_id = None
            #         move_mode = False
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if typing_mode:
        if key == 8:  # backspace
            if search_query:
                search_query = search_query[:-1]
        elif key == 13:  # enter
            query = search_query.strip().lower()
            if query == "exit":
                typing_mode = False
                search_query = ""
            elif query in SHAPE_FUNCTIONS:
                # Draw shape at current hand position
                if fx is not None and fy is not None:
                    SHAPE_FUNCTIONS[query](canvas, fx, fy)
                    print(f"Drew {query} at ({fx}, {fy})")
                search_query = ""
            elif query:
                import webbrowser
                webbrowser.open(f"https://www.google.com/search?q={query}")
                print(f"Searching Google for: {query}")
                search_query = ""
                typing_mode = False
        elif 32 <= key <= 126:  # printable chars
            search_query += chr(key)
    else:
        if key == ord('o') or key == ord('O'):
            detected_objects = detect_objects_ai(frame)
        elif key == ord('u') or key == ord('U'):
            ai_suggestions = get_drawing_suggestions()
        elif key == ord('h') or key == ord('H'):
            if fx is not None and fy is not None:
                draw_house(canvas, fx - 50, fy, 100)
        elif key == ord('s') or key == ord('S'):
            typing_mode = not typing_mode
            if not typing_mode:
                search_query = ""
        elif key == ord('g') or key == ord('G'):
            if search_query.strip():
                import webbrowser
                query = search_query.strip().lower()
                webbrowser.open(f"https://www.google.com/search?q={query}")
                print(f"Searching Google for: {query}")
        elif key == ord('-') or key == ord('_'):
            eraser_size = max(10, eraser_size - 5)
        elif key == ord('=') or key == ord('+'):
            eraser_size = min(200, eraser_size + 5)
    output = cv2.addWeighted(canvas, 0.8, frame, 1, 0)
    # show detected objects
    for obj_name, x, y, w_obj, h_obj in detected_objects:
        cv2.rectangle(output, (x, y), (x + w_obj, y + h_obj), (0, 255, 0), 2)
        cv2.putText(output, obj_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # suggestions
    if ai_suggestions:
        y0 = 50
        for sug in ai_suggestions:
            cv2.putText(output, sug, (30, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_SUCCESS, 1)
            y0 += 30
    # typing mode display
    if typing_mode:
        cv2.putText(output, f"Type query or shape name: {search_query}", (30, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_ACCENT, 2)
        cv2.putText(output, "Shapes: circle, square, rectangle, triangle, pentagon, star, ellipse, heart, diamond", (30, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_SUCCESS, 1)
        cv2.putText(output, "Type 'exit' and Enter to cancel, or Enter to search/draw", (30, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
    else:
        # Show controls help
        if youtube_mode:
            cv2.putText(output, "YOUTUBE MODE ACTIVE", (30, h - 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_SUCCESS, 2)
            cv2.putText(output, "Gestures: Index+Middle=Next | Thumb+Index+Middle=Previous | Fist=Play/Pause | 4Fingers=Scroll | Index Curl=Double-Click | Index Finger=Mouse Cursor", (30, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
            cv2.putText(output, "Open Palm=Open YouTube | Thumb+Pinky (hold)=Exit YouTube Mode", (30, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WARNING, 1)
            cv2.putText(output, "Controls: S=Search/Shapes | O=Objects | U=Suggestions | H=House | +/-=Eraser", (30, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
            cv2.putText(output, "Note: Hold gesture for smooth recognition", (30, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ACCENT, 1)
        else:
            cv2.putText(output, "Controls: S=Search/Shapes | O=Objects | U=Suggestions | H=House | +/-=Eraser", (30, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
            cv2.putText(output, "Gestures: Two fingers=Draw | Palm open=Erase | Thumb+Pinky (hold)=YouTube Mode", (30, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
            cv2.putText(output, "YouTube: Index+Middle=Next | Thumb+Index+Middle=Prev | Fist=Play/Pause | 4Fingers=Scroll | Index Curl=Double-Click | Index=Cursor", (30, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ACCENT, 1)
    cv2.putText(output, f"FPS: {fps}", (w - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ACCENT, 1)
    cv2.imshow("Gesture Drawing Pro", output)
cap.release()
cv2.destroyAllWindows()