import json
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from words.extract import WordLandmarkExtractor
from words.normalize import WordNormalizer
from sklearn.model_selection import train_test_split

DATA_DIR = Path("/content/drive/MyDrive/Sign Language Graduation Project/words/processed_data/unique_videos")
OUT_DIR = Path("/content/drive/MyDrive/Sign Language Graduation Project/words/processed_data/preprocessed_words")
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

OUT_DIR.mkdir(parents=True, exist_ok=True)

extractor = WordLandmarkExtractor(config_path="configs/words.yaml")
normalizer = WordNormalizer(config_path="configs/words.yaml")

class_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")])
label_to_idx = {d.name: idx for idx, d in enumerate(class_dirs)}

X_list = []
y_list = []
failed_videos = []

try:
    for class_dir in class_dirs:
        label_idx = label_to_idx[class_dir.name]
        videos = sorted([f for f in class_dir.iterdir() if f.suffix.lower() in VIDEO_EXTS])

        for video_path in tqdm(videos, desc=f"Class: {class_dir.name}"):
            try:
                raw_seq = extractor.process_video(str(video_path))  # (50, 378)
                clean_seq = normalizer.normalize_sequence(raw_seq)  # (50, 378)

                if np.isnan(clean_seq).any() or np.isinf(clean_seq).any():
                    failed_videos.append((str(video_path), "NaN/Inf detected"))
                    continue

                X_list.append(clean_seq)
                y_list.append(label_idx)
            except Exception as e:
                failed_videos.append((str(video_path), str(e)))
finally:
    extractor.close()

X_clean = np.asarray(X_list, dtype=np.float32)
y_labels = np.asarray(y_list, dtype=np.int64)

np.save(OUT_DIR / "X_clean.npy", X_clean)
np.save(OUT_DIR / "y_labels.npy", y_labels)
(OUT_DIR / "label_map.json").write_text(
    json.dumps(label_to_idx, ensure_ascii=False, indent=2), encoding="utf-8"
)

print("Done!")
print("X_clean shape:", X_clean.shape, "| dtype:", X_clean.dtype)
print("y_labels shape:", y_labels.shape, "| dtype:", y_labels.dtype)
print("Classes:", label_to_idx)
print("Failed count:", len(failed_videos))

#split the data
data_dir = Path("/content/drive/MyDrive/Sign Language Graduation Project/words/processed_data/preprocessed_words")

x_path = data_dir / "X_clean.npy" if (data_dir / "X_clean.npy").exists() else data_dir / "X_processed.npy"
X = np.load(x_path)
y = np.load(data_dir / "y_labels.npy")
label_map = json.loads((data_dir / "label_map.json").read_text(encoding="utf-8"))

try:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
except ValueError:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=None
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=None
    )

print("Train:", X_train.shape, y_train.shape)
print("Val:  ", X_val.shape, y_val.shape)
print("Test: ", X_test.shape, y_test.shape)
