from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2
import os

app = Flask(__name__)

# Load the trained model
model_path = r'C:\Users\anuna\OneDrive\Desktop\project\saveModel\model.keras'
model = load_model(model_path)

# Define the image size expected by the model
IMG_SIZE = (64, 64)

# Blood group labels
blood_groups = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]

def preprocess_image(image_path):
    """
    Preprocess the image for prediction
    """
    img = cv2.imread(image_path)  # Load the image using OpenCV
    if img is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")

    img_resized = cv2.resize(img, IMG_SIZE)  # Resize to (64, 64)
    img_resized = img_resized / 255.0  # Normalize pixel values
    return np.expand_dims(img_resized, axis=0)  # Add batch dimension

@app.route('/')
def index():
    return render_template('index.html')  # Render homepage template

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No file selected', 400
    
    # Save the uploaded file to a temporary directory
    upload_folder = r'C:\Users\anuna\OneDrive\Desktop\project\uploads'
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    try:
        # Preprocess the image and predict
        img_processed = preprocess_image(file_path)
        prediction = model.predict(img_processed)
        predicted_class = np.argmax(prediction, axis=1)[0]
        blood_group = blood_groups[predicted_class]

        return render_template('result.html', prediction=blood_group)  # Display result
    except Exception as e:
        return f"Error occurred during prediction: {str(e)}", 500
    finally:
        # Clean up the uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True)
