"""
Hand Gesture Controller 
"""

import cv2
import mediapipe as mp
import urllib.request
import os
import time
import collections
import math

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

try:
    import pyautogui
    pyautogui.FAILSAFE = False
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    print("[INFO] pyautogui not found — actions printed only.")

# ── Model auto-download ──
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Downloading hand landmark model (~5 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Model ready.")

# ── Tunable constants ──
HISTORY_SIZE    = 8
CONFIRM_RATIO   = 0.60
ENABLE_ACTIONS  = True
REPEAT_INTERVAL = 0.4    # seconds between repeats for volume gestures
REPEAT_GESTURES = {"Peace Sign", "Thumbs Up"}
WINDOW_NAME     = "Hand Gesture Controller"

# ── Colours (RGB) ──
GREEN  = (0,  230, 100)
CYAN   = (255,200,   0)
RED    = (50,  50, 255)
WHITE  = (255,255, 255)
BLACK  = (0,    0,   0)
ORANGE = (0,  165, 255)
PURPLE = (200,  0, 200)
GRAY   = (160,160, 160)

GESTURE_META = {
    "Open Palm":     {"label": "PLAY / PAUSE",  "color": GREEN,  "key": "playpause"},
    "Closed Fist":   {"label": "MUTE / STOP",   "color": RED,    "key": "volumemute"},
    "Peace Sign":    {"label": "VOLUME UP",      "color": CYAN,   "key": "volumeup"},
    "Thumbs Up":     {"label": "VOLUME DOWN",    "color": ORANGE, "key": "volumedown"},
    "One Finger Up": {"label": "NEXT TRACK",     "color": PURPLE, "key": "nexttrack"},
    "Unknown":       {"label": "Hold a gesture", "color": GRAY,   "key": None},
}

LEGEND = [
    ("Open Palm",     "Play / Pause  (once)"),
    ("Closed Fist",   "Mute / Stop   (once)"),
    ("Peace Sign",    "Volume Up     (hold)"),
    ("Thumbs Up",     "Volume Down   (hold)"),
    ("One Finger Up", "Next Track    (once)"),
]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

FINGER_TIPS = [4,  8, 12, 16, 20]
FINGER_PIPS = [2,  6, 10, 14, 18]
FINGER_MCPS = [2,  5,  9, 13, 17]


# ═══════════════════════
#  GESTURE CLASSIFIER
# ═══════════════════════

def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def get_finger_states(lm):
    fingers, curl_depths = [], []
    hand_size = _dist(lm[0], lm[9]) + 1e-6

    tip_to_index_mcp = _dist(lm[4], lm[5])
    tip_to_wrist     = _dist(lm[4], lm[0])
    fingers.append(tip_to_index_mcp > tip_to_wrist * 0.6)
    curl_depths.append(tip_to_index_mcp / hand_size)

    for tip, pip, mcp in zip(FINGER_TIPS[1:], FINGER_PIPS[1:], FINGER_MCPS[1:]):
        fingers.append(lm[tip].y < lm[pip].y)
        depth = (lm[tip].y - lm[mcp].y) / hand_size
        curl_depths.append(max(0.0, depth))

    return fingers, curl_depths


def classify_gesture(lm) -> str:
    f, curl = get_finger_states(lm)
    thumb, index, middle, ring, pinky = f

    hand_size = _dist(lm[0], lm[9]) + 1e-6

    # THUMB FEATURES 
    thumb_vec_y = lm[4].y - lm[2].y
    thumb_up = thumb_vec_y < -0.04  

    thumb_tip_to_index_mcp = _dist(lm[4], lm[5])
    thumb_tip_to_wrist = _dist(lm[4], lm[0])

    thumb_far = thumb_tip_to_index_mcp > hand_size * 0.55
    thumb_close = thumb_tip_to_index_mcp < hand_size * 0.38

    # Other fingers strongly curled
    fingers_curled = (
        curl[1] > 0.25 and
        curl[2] > 0.25 and
        curl[3] > 0.25 and
        curl[4] > 0.25
    )

    # 1. THUMBS UP 
    if (
        thumb_up and
        thumb_far and                  
        fingers_curled and
        not index and not middle and not ring and not pinky
    ):
        return "Thumbs Up"

    # 2. CLOSED FIST 
    if (
        not index and not middle and not ring and not pinky
        and fingers_curled
        and not thumb_up              
        and thumb_close               
    ):
        return "Closed Fist"

    # 3. PEACE SIGN
    if (
        index and middle
        and not ring and not pinky
        and curl[3] > 0.2 and curl[4] > 0.2
    ):
        return "Peace Sign"

    # 4. ONE FINGER
    if (
        index
        and not middle and not ring and not pinky
        and curl[2] > 0.2 and curl[3] > 0.2
    ):
        return "One Finger Up"

    # 5. OPEN PALM
    if sum(f) >= 4:
        return "Open Palm"

    return "Unknown"


# ════════════
#  ACTIONS
# ════════════

def perform_action(gesture: str):
    meta = GESTURE_META[gesture]
    print(f"[ACTION] {meta['label']}")
    if not ENABLE_ACTIONS or not PYAUTOGUI_AVAILABLE:
        return
    if meta["key"]:
        pyautogui.press(meta["key"])


class ActionController:
    def __init__(self):
        self.last_gesture   = None
        self.last_fire_time = 0.0

    def process(self, confirmed: str, now: float):
        if confirmed == "Unknown":
            self.last_gesture = None
            return

        elapsed = now - self.last_fire_time

        if confirmed in REPEAT_GESTURES:
            # Fire immediately on first see, then every REPEAT_INTERVAL while held
            if confirmed != self.last_gesture or elapsed >= REPEAT_INTERVAL:
                perform_action(confirmed)
                self.last_gesture   = confirmed
                self.last_fire_time = now
        else:
            # Fire once when the gesture first appears
            if confirmed != self.last_gesture:
                perform_action(confirmed)
                self.last_gesture   = confirmed
                self.last_fire_time = now


# ════════════
#  DRAWING
# ════════════

def draw_landmarks(frame, hand_landmarks_list):
    h, w = frame.shape[:2]
    pts  = None
    for landmarks in hand_landmarks_list:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        pad = 18
        cv2.rectangle(frame,
                      (max(0, min(xs)-pad), max(0, min(ys)-pad)),
                      (min(w, max(xs)+pad), min(h, max(ys)+pad)),
                      CYAN, 1, cv2.LINE_AA)
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], GRAY, 2, cv2.LINE_AA)
        for i, pt in enumerate(pts):
            color = GREEN if i in FINGER_TIPS else WHITE
            cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 5, BLACK,  1, cv2.LINE_AA)
    return pts


def draw_gesture_near_hand(frame, pts, raw_gesture):
    if pts is None or raw_gesture == "Unknown":
        return
    xs, ys  = [p[0] for p in pts], [p[1] for p in pts]
    label_x = min(xs)
    label_y = max(0, min(ys) - 28)
    meta    = GESTURE_META.get(raw_gesture, GESTURE_META["Unknown"])
    cv2.putText(frame, raw_gesture, (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, meta["color"], 2, cv2.LINE_AA)


def draw_legend(frame):
    h, w    = frame.shape[:2]
    line_h  = 22
    padding = 10
    box_w   = 265
    box_h   = len(LEGEND) * line_h + padding * 2
    ov      = frame.copy()
    cv2.rectangle(ov, (w - box_w - 10, h - box_h - 10),
                      (w - 10, h - 10), BLACK, -1)
    cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
    for i, (gesture, action) in enumerate(LEGEND):
        y     = h - box_h - 10 + padding + i * line_h + 14
        color = GESTURE_META[gesture]["color"]
        cv2.putText(frame, f"{gesture:<14} {action}", (w - box_w - 4, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def draw_hud(frame, raw, confirmed, confidence, fps, hand_count):
    h, w = frame.shape[:2]
    meta = GESTURE_META.get(confirmed, GESTURE_META["Unknown"])

    ov = frame.copy()
    cv2.rectangle(ov, (10, 10), (w - 10, 65), BLACK, -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, meta["label"], (25, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, meta["color"], 2, cv2.LINE_AA)

    if raw != "Unknown" and raw != confirmed:
        bar_w     = int((w - 20) * min(confidence / CONFIRM_RATIO, 1.0))
        bar_color = GESTURE_META.get(raw, GESTURE_META["Unknown"])["color"]
        cv2.rectangle(frame, (10, 70), (10 + bar_w, 82), bar_color, -1)
        cv2.rectangle(frame, (10, 70), (w - 10,     82), GRAY,      1)
        cv2.putText(frame, f"Detecting: {raw}", (12, 96),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bar_color, 1, cv2.LINE_AA)

    ov2 = frame.copy()
    cv2.rectangle(ov2, (10, h - 75), (200, h - 10), BLACK, -1)
    cv2.addWeighted(ov2, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, f"FPS: {fps:.1f}",      (20, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Hands: {hand_count}", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, CYAN,  1, cv2.LINE_AA)

    draw_legend(frame)


# ════════════
#  SMOOTHER
# ════════════

class GestureSmoother:
    def __init__(self, n=HISTORY_SIZE):
        self.history = collections.deque(maxlen=n)

    def update(self, g: str):
        self.history.append(g)
        counts = collections.Counter(self.history)
        counts.pop("Unknown", None)
        if not counts:
            return "Unknown", 0.0
        top_gesture, top_count = counts.most_common(1)[0]
        confidence = top_count / len(self.history)
        if confidence >= CONFIRM_RATIO:
            return top_gesture, confidence
        return "Unknown", confidence

    def reset(self):
        self.history.clear()


# ════════
#  MAIN
# ════════

def main():
    ensure_model()
    print("=" * 48)
    print("  HAND GESTURE CONTROLLER — Ready")
    print("  Peace Sign / Thumbs Up: hold to repeat")
    print("  Q or ESC to quit")
    print("=" * 48)

    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    smoother          = GestureSmoother()
    action_ctrl       = ActionController()
    fps               = 0.0
    prev_time         = time.time()
    start_time        = time.time()
    confirmed_gesture = "Unknown"

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame     = cv2.flip(frame, 1)
            now       = time.time()
            dt        = max(now - prev_time, 1e-6)
            prev_time = now
            fps       = 0.9 * fps + 0.1 * (1.0 / dt)

            rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((now - start_time) * 1000)

            result     = landmarker.detect_for_video(mp_image, timestamp_ms)
            raw        = "Unknown"
            hand_count = 0
            pts        = None

            if result.hand_landmarks:
                hand_count = len(result.hand_landmarks)
                lm         = result.hand_landmarks[0]
                pts        = draw_landmarks(frame, result.hand_landmarks)
                raw        = classify_gesture(lm)
            else:
                smoother.reset()
                confirmed_gesture        = "Unknown"
                action_ctrl.last_gesture = None

            confirmed, confidence = smoother.update(raw)

            if confirmed != "Unknown":
                confirmed_gesture = confirmed

            action_ctrl.process(confirmed, now)

            draw_gesture_near_hand(frame, pts, raw)
            draw_hud(frame, raw, confirmed_gesture, confidence, fps, hand_count)
            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done. Goodbye!")


if __name__ == "__main__":
    main()