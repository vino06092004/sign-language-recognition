# 🤟 American Sign Language (ASL) Recognition System

A real-time American Sign Language recognition web application powered by a Convolutional Neural Network (CNN) and Streamlit. The system recognizes 29 ASL hand gestures (A–Z, space, delete, and nothing) from webcam snapshots or image uploads with over 98% validation accuracy.

---

## 📌 Features
- **Real-Time Classification:** Snap photos directly using your webcam or upload existing gesture images.
- **Top-3 Match Probabilities:** Displays confidence scores and alternative gesture candidates.
- **High Accuracy & Robustness:** Trained on 87,000 images using a custom CNN architecture with data augmentation and batch normalization to handle lighting and background variations.

---

## 🛠️ Tech Stack
- **Deep Learning:** TensorFlow / Keras
- **Web App / UI:** Streamlit
- **Image Processing:** OpenCV, Pillow, NumPy
- **Dataset:** ASL Alphabet (29 Classes)

---

## 🚀 Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/vino06092004/sign-language-recognition.git](https://github.com/vino06092004/sign-language-recognition.git)
   cd sign-language-recognition

   Create and activate a virtual environment:

PowerShell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
Install dependencies:

Bash
pip install -r requirements.txt
Launch the web application:

Bash
streamlit run app.py
🧠 Model Architecture & Training Highlights
Input Resolution: 64x64 RGB images

Layers: 3 Convolutional Blocks (Conv2D + BatchNormalization + MaxPooling2D) followed by a Dense layer with 40% Dropout to prevent overfitting.

Data Augmentation: Random rotations, translations, zooming, and contrast adjustments.

Validation Accuracy: 98.03%

Loss Function: Categorical Crossentropy | Optimizer: Adam



