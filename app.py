from flask import Flask, request, jsonify
import joblib
import base64
from utils import highlight_offensive_words, extract_hog_features

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained models & transformers
text_model   = joblib.load("models/text_model.pkl")
vectorizer   = joblib.load("models/tfidf_vectorizer.pkl")
image_clf    = joblib.load("models/image_model.pkl")
image_scaler = joblib.load("models/image_scaler.pkl")


@app.route("/", methods=["GET"])
def home():
    """Health‐check endpoint."""
    return "✅ Flask API is up!"


@app.route("/detect_text", methods=["POST"])
def detect_text():
    """
    Expects JSON: { "text": "some user input" }
    Returns JSON: { "prediction_class": int, "flagged_words": [ ... ] }
    """
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Vectorize and predict
    X = vectorizer.transform([text])
    pred_class = int(text_model.predict(X)[0])

    # Highlight any offensive words
    flagged = highlight_offensive_words(text)

    return jsonify({
        "prediction_class": pred_class,
        "flagged_words": flagged
    })


@app.route("/detect_image", methods=["POST"])
def detect_image():
    """
    Expects JSON: { "image": "<base64‐encoded image>" }
    Returns JSON:
      {
        "is_harmful": bool,
        "confidence": float,
        "harmful_type": str or null,
        "explanation": str or null
      }
    """
    data = request.get_json(force=True)
    img_b64 = data.get("image", "").strip()
    if not img_b64:
        return jsonify({"error": "No image provided"}), 400

    try:
        # Decode base64, extract HOG features, scale, and predict
        img_bytes   = base64.b64decode(img_b64)
        feat        = extract_hog_features(img_bytes)
        feat_scaled = image_scaler.transform([feat])
        pred        = int(image_clf.predict(feat_scaled)[0])
        score       = float(image_clf.decision_function(feat_scaled)[0])
    except Exception as e:
        return jsonify({"error": f"Processing error: {e}"}), 500

    is_harmful  = bool(pred == 1)
    harmful_type = "⚠️ Harmful Visual Content Detected" if is_harmful else None
    explanation = (
    "AI detected visual patterns that statistically align with unsafe or inappropriate content — could be alcohol, weapons, or something dystopian. Viewer discretion advised."
    if is_harmful else None
)


    return jsonify({
        "is_harmful":   is_harmful,
        "confidence":   score,
        "harmful_type": harmful_type,
        "explanation":  explanation
    })


if __name__ == "__main__":
    # Print registered routes for sanity check
    print("Registered routes:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule}")
    app.run(debug=True)
