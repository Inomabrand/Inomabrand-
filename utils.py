# utils.py
import re
from io import BytesIO
from PIL import Image
import numpy as np
from skimage.transform import resize
from skimage.feature import hog

# Expanded dictionaries for flagging text
SLURS = {
    "bitch", "slut", "fucker", "hoe", "bastard", "asshole", "dick", "douche",
    "cunt", "pussy", "whore", "jerk", "fhate", "faggot",
    "negro", "nigger", "retard", "spastic", "tranny", "cracker"
}

# Phrases and words indicating threats or violent content
VIOLENT_PHRASES = [
    "hope you die", "kill yourself", "i hope you die", "i hope you and your kid die",
    "kill you", "i will kill you", "you are dead", "you're dead",
    "die in an accident", "go to hell", "fuck off", "eat shit",
    "go die", "self harm", "hang yourself", "shoot you",
    "stab you", "hurt you", "punch you", "murder you", "massacre",
    "die", "kill", "accident"
]


def highlight_offensive_words(text: str) -> list:
    """
    Returns a list of offensive or threatening phrases/words found in the text.
    Flags both specific slurs and violent/threatening expressions.
    """
    text_lower = text.lower()
    flagged = set()

    # Check for multi-word violent phrases first
    for phrase in VIOLENT_PHRASES:
        if phrase in text_lower:
            flagged.add(phrase)

    # Tokenize and check for slurs and single-word threats
    tokens = re.findall(r"\w+", text_lower)
    for token in tokens:
        if token in SLURS or token in {"die", "kill", "accident", "murder"}:
            flagged.add(token)

    return list(flagged)


def extract_hog_features(image_bytes: bytes,
                         img_size=(64, 64),
                         orientations=9,
                         pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2)) -> np.ndarray:
    """
    Extracts HOG features from raw image bytes.
    """
    # Load and convert to grayscale
    img = Image.open(BytesIO(image_bytes)).convert("L")
    img = resize(np.array(img), img_size)

    features = hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm='L2-Hys'
    )
    return features
