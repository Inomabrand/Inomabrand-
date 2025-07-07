🔍 AI-Powered Content Moderation System

A Python-based AI solution for detecting offensive language and harmful imagery in social media posts. This tool uses natural language processing (NLP) and computer vision techniques to classify and flag inappropriate content, helping platforms maintain a safer online community.

🚀 Features

✅ Detects **offensive language**, **hate speech**, and **toxic comments** using NLP.
🖼️ Flags **harmful imagery** such as violence or adult content using deep learning.
📦 Easy-to-use **Flask API** for integration with web or mobile platforms.
🖥️ Includes a **Streamlit dashboard** for interactive testing and visualization.


🧠 Tech Stack

NLP: scikit-learn, NLTK / spaCy, TF-IDF
Vision: PyTorch / TensorFlow, OpenCV, pre-trained CNNs (e.g., ResNet)
Backend: Flask
Frontend: Streamlit
Other tools: joblib, PIL, requests

⚙️ Installation


Clone the repository
git clone https://github.com/Inomabrand/Inomabrand-.git
cd content-moderation-ai

Create virtual environment
python -m venv env
source env/bin/activate  # or `env\Scripts\activate` on Windows

Install dependencies
pip install -r requirements.txt


🏃‍♂️ Usage

Start the Flask API

"python app.py"

Launch the Streamlit Dashboard

"streamlit run dashboard.py"

📊 Model Training (Optional)

If you want to retrain the models:

"For text model
python train_text_model.py"

"For image model
python train_image_model.py
"


📷 Harmful Image Detection

Upload an image via the dashboard or call the API  endpoint with base64-encoded images.

🛡️ Label Categories

Text:

Hate Speech
   Offensive Language
   No Offensive Content
Image:

 Harmful
 Safe

📄 License

This project is licensed under the MIT License.

🤝 Contributing

Pull requests and issues are welcome. For major changes, please open an issue first to discuss what you would like to change.
