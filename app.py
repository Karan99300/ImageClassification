from flask import Flask, request, jsonify, render_template
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import requests
import torch

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model and feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained('karan99300/ConvNext-finetuned-CIFAR100')
model = AutoModelForImageClassification.from_pretrained('karan99300/ConvNext-finetuned-CIFAR100')

# Define route for home page with form
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get image URL from form submission
        image_url = request.form['image_url']
        
        # Classify image
        predicted_class = classify_image(image_url)
        
        return render_template('index.html', predicted_class=predicted_class, image_url=image_url)
    
    return render_template('index.html')

# Function to classify image
def classify_image(image_url):
    # Fetch image from URL
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except Exception as e:
        return f'Error fetching image: {str(e)}'
    
    # Preprocess image and perform inference
    pixel_values = feature_extractor(image.convert('RGB'), return_tensors='pt').pixel_values
    with torch.no_grad():
        outputs = model(pixel_values)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
    
    # Get predicted label
    predicted_label = model.config.id2label[predicted_class_idx]
    
    return predicted_label

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True,port=5000)
