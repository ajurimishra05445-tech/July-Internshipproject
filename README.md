# Handwritten Digit Recognizer
A full-stack Machine learning web application built using **Flask** , **TensorFlow**, and an interactive **HTML5 Canvas**. Users can draw any digit from 0 to 9 on the canvas ,and the underlying Deep Learning model will predict the drawn digit in real-time.
## Features
* **Interactive Canvas:**
* **Real-time Inference:** Process base64 canvas data and get fast predictions.
* **Preprocessing Pipeline:** Converts input to 28x28 grayscale image arrays matching the MNIST standard.

## 🛠️ Project Structure
```
digit-recognizer/
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── app.py
├── mnist_model.keras
├── requirements.txt
├── LICENSE
└── README.md
