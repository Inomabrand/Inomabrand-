import streamlit as st
import requests
import base64

# Mapping numeric class to human-readable labels
LABEL_MAP = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "No Offensive Content"
}

API_URL = "http://127.0.0.1:5000"

st.set_page_config(page_title="Content Moderation Dashboard", layout="wide")

st.title("🛡 Content Moderation Dashboard")

# --- Text Moderation Section ---
st.header("🔤 Text Moderation")
user_text = st.text_area("Enter text to analyze:", height=150)

if st.button("Analyze Text"):
    if not user_text.strip():
        st.error("Please enter some text to analyze.")
    else:
        payload = {"text": user_text}
        try:
            response = requests.post(f"{API_URL}/detect_text", json=payload)
            if response.status_code == 200:
                result = response.json()
                pred = result.get("prediction_class")
                label = LABEL_MAP.get(pred, "Unknown")
                # Display result
                st.markdown(f"**Classification:** {label}")

                # Flagged words
                flagged = result.get("flagged_words", [])
                if flagged:
                    st.markdown("**Flagged Keywords:**")
                    st.markdown(
                        """
                        <div style='background-color:#ffdddd; padding:10px; border-radius:5px;'>
                        """ + ", ".join(f"<code>{w}</code>" for w in flagged) + "</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown("**Flagged Keywords:** None detected.")
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Exception: {e}")

st.markdown("---")

# --- Image Moderation Section ---
st.header("🖼️ Image Moderation")
uploaded_file = st.file_uploader("Upload an image to analyze", type=["jpg", "jpeg", "png"])
if st.button("Analyze Image"):
    if uploaded_file is None:
        st.error("Please upload an image first.")
    else:
        image_bytes = uploaded_file.read()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        payload = {"image": encoded_image}
        try:
            response = requests.post(f"{API_URL}/detect_image", json=payload)
            if response.status_code == 200:
                result = response.json()
                is_harmful = result.get("is_harmful")
                confidence = result.get("confidence")
                st.markdown(f"**Harmful Content Detected:** {is_harmful}")
                st.markdown(f"**Confidence Score:** {confidence:.2f}")
                if is_harmful:
                    st.markdown(f"**Type:** {result.get('harmful_type')} ")
                    st.markdown(f"**Explanation:** {result.get('explanation')}")
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Exception: {e}")
