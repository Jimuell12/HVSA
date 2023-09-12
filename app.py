from flask import Flask, render_template, request, jsonify, redirect
import torch
import io
from PyPDF2 import PdfReader
from docx import Document
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import regex
import numpy as np
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)

model_path = "rimuruu1/TextDetection"  # Update with your model path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/privacy-policy')
def privacy_policy():
    return render_template('pp.html')

@app.route('/terms-of-service')
def terms_of_service():
    return render_template('tos.html')

def extractpdf(file):
    paragraphs = []
    try:
        pdf_reader = PdfReader(file)
        num_pages = len(pdf_reader.pages)

        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            paragraphs.append(page.extract_text())
        

    except Exception as e:
        print("An error occurred while extracting PDF:", str(e))

    return paragraphs

def extractdocx(file):
    paragraphs = []
    try:
        doc = Document(file)
        for paragraph in doc.paragraphs:
            paragraphs.append(paragraph.text)
        
    except Exception as e:
        print("An error occurred while extracting DOCX:", str(e))
    return paragraphs

def extractimage(file):
    image = np.array(Image.open(file))
    paragraphs = pytesseract.image_to_string(image)
    
    return paragraphs

def predict(paragraphs):
    predictions = []

    for paragraph in paragraphs:
        inputs = tokenizer(paragraph, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)
        
        ai_probability = probabilities[0][1].item()
        human_probability = probabilities[0][0].item()

        predictions.append({
            "ai_probability": ai_probability,
            "human_probability": human_probability,
            "text": paragraph
        })

    return predictions

def predict2(paragraph):
    predictions = []

    inputs = tokenizer(paragraph, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)
        
    ai_probability = probabilities[0][1].item()
    human_probability = probabilities[0][0].item()

    predictions.append({
            "ai_probability": ai_probability,
            "human_probability": human_probability,
            "text": paragraph
    })

    return predictions

@app.route('/upload', methods=['POST'])
def upload():
    file_content = request.files['file']

    if file_content.filename.endswith('.pdf'):
        paragraphs = extractpdf(file_content)
        predictions = predict(paragraphs)

        # Compute average probabilities
        total_ai_probability = sum(prediction["ai_probability"] for prediction in predictions)
        total_human_probability = sum(prediction["human_probability"] for prediction in predictions)
        avg_ai_probability = total_ai_probability / len(predictions)
        avg_human_probability = total_human_probability / len(predictions)

        return render_template(
            'index.html',
            avg_ai_probability=avg_ai_probability,
            avg_human_probability=avg_human_probability,
            predictions=predictions
        )
    elif file_content.filename.endswith(".docx"):
        paragraphs = extractdocx(file_content)
        predictions = predict(paragraphs)

        # Compute average probabilities
        total_ai_probability = sum(prediction["ai_probability"] for prediction in predictions)
        total_human_probability = sum(prediction["human_probability"] for prediction in predictions)
        avg_ai_probability = total_ai_probability / len(predictions)
        avg_human_probability = total_human_probability / len(predictions)


        return render_template(
            'index.html',
            avg_ai_probability=avg_ai_probability,
            avg_human_probability=avg_human_probability,
            predictions=predictions
        )
    
    elif file_content.filename.endswith(".png") or file_content.filename.endswith(".jpg"):
        paragraphs = extractimage(file_content)
        predictions = predict2(paragraphs)

        # Compute average probabilities
        total_ai_probability = sum(prediction["ai_probability"] for prediction in predictions)
        total_human_probability = sum(prediction["human_probability"] for prediction in predictions)
        avg_ai_probability = total_ai_probability / len(predictions)
        avg_human_probability = total_human_probability / len(predictions)

        return render_template(
            'index.html',
            avg_ai_probability=avg_ai_probability,
            avg_human_probability=avg_human_probability,
            predictions=predictions
        )
    else:
        return render_template('index.html', error="Uploaded file must be a PDF or DOCX.")

@app.route('/predict_text', methods=['POST'])
def textpredict():
    text_input = request.form['text-input']
    if text_input:
        predictions = predict2(text_input)

        # Compute average probabilities
        total_ai_probability = sum(prediction["ai_probability"] for prediction in predictions)
        total_human_probability = sum(prediction["human_probability"] for prediction in predictions)
        avg_ai_probability = total_ai_probability / len(predictions)
        avg_human_probability = total_human_probability / len(predictions)

        return render_template(
            'index.html',
            avg_ai_probability=avg_ai_probability,
            avg_human_probability=avg_human_probability,
            predictions=predictions
        )
    else:
        return render_template('index.html', error="No input")


app.static_folder = 'static'

if __name__ == '__main__':
    app.run(debug=True)
