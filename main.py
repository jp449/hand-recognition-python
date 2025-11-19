from google.colab.output import eval_js
from IPython.display import display, Javascript
import cv2
import numpy as np
import PIL.Image
import io
import base64
from google.colab.patches import cv2_imshow
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3

js = Javascript('''
    async function captureImage() {
        const video = document.createElement('video');
        document.body.appendChild(video);
        const stream = await navigator.mediaDevices.getUserMedia({video: true});
        video.srcObject = stream;
        await new Promise((resolve) => video.onloadedmetadata = resolve);
        video.play();

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);

        stream.getTracks().forEach(track => track.stop());
        video.remove();

        return canvas.toDataURL('image/jpeg');
    }
''')

def capture_frame():
    display(js)
    data = eval_js("captureImage()")
    _, encoded = data.split(',', 1)
    image_bytes = base64.b64decode(encoded)
    image = PIL.Image.open(io.BytesIO(image_bytes))
    return np.array(image)


def count_fingers(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    fingers_up = 0
    landmarks = hand_landmarks.landmark

    for tip in finger_tips:
        if landmarks[tip].y < landmarks[tip - 2].y:
            fingers_up += 1

    return fingers_up


def detect_thumb(hand_landmarks):
    landmarks = hand_landmarks.landmark
    if landmarks[4].y < landmarks[1].y:
        return 1
    return 0

print("Please run the code and show your hand to the camera.")
frame = capture_frame()

frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
frame_resized = cv2.resize(frame, (640, 480))
results = hands.process(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))

if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        mp_draw.draw_landmarks(frame_resized, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        fingers_up = count_fingers(hand_landmarks)
        thumb_up = detect_thumb(hand_landmarks)

        # Display finger count on the frame
        cv2.putText(frame_resized, f'Fingers: {fingers_up}', (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        if thumb_up == 1:
            cv2.putText(frame_resized, 'Thumb: 1', (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    print(f"Detected Fingers , Thumb: {fingers_up},{thumb_up}")
else:
    print("No hands detected. Try again.")

cv2_imshow(frame_resized)

