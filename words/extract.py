import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from types import SimpleNamespace
import urllib.request
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import ymal

class WordLandmarkExtractor:
    
    def __init__(self,
        config_path: str = "configs/words.yaml",
        num_frames: int | None = None,
        model_complexity: int | None = None,
        model_asset_path: str = "/content/holistic_landmarker.task",
                ):

        cfg_file = Path(config_path)
        full_cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) if cfg_file.exists() else {}
        data_cfg = full_cfg.get("data", {})
        ext_cfg = full_cfg.get("extraction", {})
        self.num_frames = int(num_frames if num_frames is not None else data_cfg.get("num_frames", 50))
        self.model_complexity = int(
            model_complexity if model_complexity is not None else ext_cfg.get("model_complexity", 1)
        )
        self.num_pose = int(ext_cfg.get("pose_landmarks", 33))
        self.num_face = int(ext_cfg.get("face_landmarks", 40))
        self.num_hand = int(ext_cfg.get("hand_landmarks", 21))

        model_path = Path(ext_cfg.get("model_asset_path", model_asset_path)
        model_url = ext_cfg.get("model_url", self.DEFAULT_MODEL_URL)
        )

    @staticmethod
    def _points(landmarks):
        
        if not landmarks:
            return []
        if hasattr(landmarks, "landmark"):
            return landmarks.landmark
        if isinstance(landmarks[0], (list, tuple)):
            return landmarks[0]
        return landmarks

    def extract_landmarks(self, results):
        pose = np.zeros(self.num_pose * 4, dtype=np.float32)
        face = np.zeros(self.num_face, dtype=np.float32)
        left_hand = np.zeros(self.num_hand * 3, dtype=np.float32)
        right_hand = np.zeros(self.num_hand * 3, dtype=np.float32)

        if self._points(results.pose_landmarks):
            pose = np.array([
                value
                for lm in self._points(results.pose_landmarks)
                for value in (lm.x, lm.y, lm.z, lm.visibility)
            ], dtype=np.float32)

        if self._points(results.face_landmarks):
            face_points = self._points(results.face_landmarks)[:self.num_face]
            face = np.array([
                value
                for lm in face_points
                for value in (lm.x, lm.y, lm.z)
            ], dtype=np.float32)

        if self._points(results.left_hand_landmarks):
            left_hand = np.array([
                value
                for lm in self._points(results.left_hand_landmarks)
                for value in (lm.x, lm.y, lm.z)
            ], dtype=np.float32)

        if self._points(results.right_hand_landmarks):
            right_hand = np.array([
                value
                for lm in self._points(results.right_hand_landmarks)
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
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = self.landmarker.detect(mp_image)

                sequence.append(
                    self.extract_landmarks(results)
                )
        finally:
            cap.release()

        return np.array(sequence, dtype=np.float32)

    def close(self):
        self.landmarker.close()
