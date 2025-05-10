from django.shortcuts import render, redirect
from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import logging
from .models import Prediction

# Setup detailed logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Log TensorFlow version for compatibility check
logging.info(f"TensorFlow version: {tf.__version__}")
MODEL_PATH = r"D:/vinay/programming codeas apps/SkinCancerDetection/skin_cancer_model.keras" 
try:
    model = load_model(MODEL_PATH)
    input_shape = model.input_shape
    output_shape = model.output_shape
    logging.info(f"Model input shape: {input_shape}")
    logging.info(f"Model output shape: {output_shape}")
    if input_shape[1:3] != (128, 128) or output_shape[1] != 9:
        logging.warning("Model input/output shapes may not match training configuration!")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise

class_names = [
    'pigmented benign keratosis', 'melanoma', 'vascular lesion', 'actinic keratosis',
    'squamous cell carcinoma', 'basal cell carcinoma', 'seborrheic keratosis',
    'dermatofibroma', 'nevus'
]

@login_required
def home(request):
    return render(request, 'main/home.html')

def signup(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')
    else:
        form = UserCreationForm()
    return render(request, 'main/signup.html', {'form': form})

@login_required
def predict_disease(request):
    prediction = None
    confidence = None
    probabilities = None
    error = None
    image_url = None

    if request.method == 'POST' and request.FILES.get('image'):
        try:
            uploaded_image = request.FILES['image']
            valid_extensions = ['.jpg', '.jpeg', '.png']
            if not any(uploaded_image.name.lower().endswith(ext) for ext in valid_extensions):
                error = "Invalid image format. Please upload a JPG, JPEG, or PNG file."
                logging.error(error)
                return render(request, 'main/predict.html', {
                    'prediction': prediction,
                    'confidence': confidence,
                    'probabilities': probabilities,
                    'error': error,
                    'image_url': image_url
                })

            # Load and preprocess image
            try:
                img = Image.open(uploaded_image).convert("RGB")
                logging.info(f"Image mode: {img.mode}, size: {img.size}")
                img = img.resize((128, 128), Image.Resampling.LANCZOS)  # Use LANCZOS for high-quality resizing
                img_array = np.array(img, dtype=np.uint8)  # Ensure uint8 for consistency with training
                logging.info(f"Image array shape: {img_array.shape}, min: {img_array.min()}, max: {img_array.max()}")
                img_array = np.expand_dims(img_array, axis=0) 
            except Exception as e:
                logging.error(f"Image processing error: {e}")
                error = "Failed to process the image. Please upload a valid image file."
                return render(request, 'main/predict.html', {
                    'prediction': prediction,
                    'confidence': confidence,
                    'probabilities': probabilities,
                    'error': error,
                    'image_url': image_url
                })

            
            try:
                preds = model.predict(img_array, verbose=0)
                prediction = class_names[np.argmax(preds[0])]
                confidence = 100 * np.max(preds[0])
                probabilities = {class_names[i]: f"{preds[0][i]*100:.2f}%" for i in range(len(class_names))}

                logging.info(f"Raw probabilities: {preds[0]}")
                logging.info(f"Predicted class: {prediction}")
                logging.info(f"Confidence: {confidence:.2f}%")
            except Exception as e:
                logging.error(f"Model prediction error: {e}")
                error = "An error occurred during prediction. Please try again."
                return render(request, 'main/predict.html', {
                    'prediction': prediction,
                    'confidence': confidence,
                    'probabilities': probabilities,
                    'error': error,
                    'image_url': image_url
                })

            try:
                uploaded_image.seek(0)  
                prediction_obj = Prediction(
                    user=request.user,
                    image=uploaded_image,
                    prediction=prediction,
                    confidence=confidence
                )
                prediction_obj.save()
                image_url = prediction_obj.image.url 
            except Exception as e:
                logging.error(f"Database save error: {e}")
                error = "Failed to save prediction to database. Please try again."
                return render(request, 'main/predict.html', {
                    'prediction': prediction,
                    'confidence': confidence,
                    'probabilities': probabilities,
                    'error': error,
                    'image_url': image_url
                })


        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            error = "An unexpected error occurred. Please try again with a valid image."

    return render(request, 'main/predict.html', {
        'prediction': prediction,
        'confidence': confidence,
        'probabilities': probabilities,
        'error': error,
        'image_url': image_url
    })

@login_required
def diseases_info(request):
    return render(request, 'main/disease_info.html')