# train_image.py
import os
import pandas as pd
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
from PIL import Image

# --- PARAMETERS ---
IMG_SIZE = (64, 64)           # much smaller => 4× fewer pixels
ORIENTATIONS = 9
PIXELS_PER_CELL = (8, 8)
CELLS_PER_BLOCK = (2, 2)
N_JOBS = -1                   # use all cores
# -------------------

# 1) Load annotations
df = pd.read_csv("data/harmful_images.csv")
df['full_path'] = df['imagePath'].apply(lambda fn: os.path.join("data/images", fn))
df['label'] = df['decision'].str.lower().eq('yes').astype(int)

# 2) Feature‐extraction worker
def process_row(path, label):
    try:
        img = imread(path, as_gray=True)
    except Exception:
        img = np.array(Image.open(path).convert('L'))
    img = resize(img, IMG_SIZE)
    hog_vec = hog(
        img,
        orientations=ORIENTATIONS,
        pixels_per_cell=PIXELS_PER_CELL,
        cells_per_block=CELLS_PER_BLOCK,
        block_norm='L2-Hys'
    )
    return hog_vec, label

# 3) Parallel HOG extraction
print(f"Extracting HOG features from {len(df)} images…")
results = Parallel(n_jobs=N_JOBS)(
    delayed(process_row)(r.full_path, r.label)
    for r in tqdm(df.itertuples(), total=len(df))
)

# 4) Build feature matrix
features, labels = zip(*results)
X = np.stack(features)
y = np.array(labels)

# 5) Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 6) Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

# 7) Train a fast SGD SVM
print("Training SGDClassifier (linear SVM)…")
clf = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, n_jobs=-1)
clf.fit(X_train, y_train)

# 8) Evaluate
acc = clf.score(X_val, y_val)
print(f"Validation accuracy: {acc:.4f}")

# 9) Save models
os.makedirs("models", exist_ok=True)
joblib.dump(clf,      "models/image_model.pkl")
joblib.dump(scaler,   "models/image_scaler.pkl")
print("Saved image_model.pkl and image_scaler.pkl")
