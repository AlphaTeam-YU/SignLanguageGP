import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from types import SimpleNamespace
import urllib.request
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import yaml

class WordLandmarkExtractor:
    
    def __init__(self, num_frames=50 , model_complexity=1):
        self.num_frames = num_frames
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity
        )

    def extract_landmarks(self, results):
        pose = np.zeros(33 * 4, dtype=np.float32)
        face = np.zeros(40 * 3, dtype=np.float32)
        left_hand = np.zeros(21 * 3, dtype=np.float32)
        right_hand = np.zeros(21 * 3, dtype=np.float32)

        if results.pose_landmarks:
            pose = np.array([
                value
                for lm in results.pose_landmarks.landmark
                for value in (lm.x, lm.y, lm.z, lm.visibility)
            ], dtype=np.float32)

        if results.face_landmarks:
            face_points = results.face_landmarks.landmark[:40]
            face = np.array([
                value
                for lm in face_points
                for value in (lm.x, lm.y, lm.z)
            ], dtype=np.float32)

        if results.left_hand_landmarks:
            left_hand = np.array([
                value
                for lm in results.left_hand_landmarks.landmark
                for value in (lm.x, lm.y, lm.z)
            ], dtype=np.float32)

        if results.right_hand_landmarks:
            right_hand = np.array([
                value
                for lm in results.right_hand_landmarks.landmark
                for value in (lm.x, lm.y, lm.z)
            ], dtype=np.float32)

        return np.concatenate([
            pose, face, left_hand, right_hand
        ])

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Invalid frame count: {video_path}")

        indices = np.linspace(
            0, total_frames - 1, self.num_frames
        ).astype(int)

        sequence = []

        try:
            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                success, frame = cap.read()

                if not success:
                    raise ValueError(
                        f"Cannot read frame {index}: {video_path}"
                    )

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.holistic.process(rgb)

                sequence.append(
                    self.extract_landmarks(results)
                )
        finally:
            cap.release()

        return np.array(sequence, dtype=np.float32)

    def close(self):
        self.holistic.close()
