# Fake News Detection in Tamil and English

## Overview
This project focuses on detecting fake news in both Tamil and English languages using Transformer-based NLP models like DistilBERT, ALBERT, DistilMBERT, and IndicBERT. These models are fine-tuned to classify news articles as either real or fake, helping users verify the authenticity of news sources.

## Features
- Supports fake news detection in both Tamil and English
- Utilizes Transformer-based models for better accuracy
- Pretrained models fine-tuned for domain-specific classification
- User-friendly web interface for real-time news verification using Streamlit

## Dataset
The dataset used for training consists of labeled Tamil and English news articles collected from multiple sources. The dataset contains 16,527 English news articles and 5,227 Tamil news articles across various domains like politics, healthcare, sports, cinema, and COVID-19. Preprocessing includes text normalization, stopword removal, and tokenization.

## Technologies Used
- Python
- Natural Language Processing (NLTK, spaCy, Transformers)
- Deep Learning (PyTorch, TensorFlow, Hugging Face Transformers)
- Machine Learning (Scikit-learn)
- Streamlit (for UI visualization and deployment)

## Installation
To set up the project, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/ShivaniKrishnaKumar/Fake_News_Detection.git
   ```
2. Navigate to the project directory:
   ```sh
   cd Fake_News_Detection
   ```
3. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   venv\Scripts\activate # On Mac: source venv/bin/activate
   ```
4. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Model Performance
The following Transformer-based models were used for classification:

**English Models:**
- **DistilBERT** - Accuracy: 85.69%
- **ALBERT** - Accuracy: 85.14%

**Tamil Models:**
- **DistilMBERT** - Accuracy: 82.02%
- **IndicBERT** - Accuracy: 75.81%

## Deployment
The model is deployed as a web application using Streamlit, allowing users to submit news articles for verification. The web interface classifies the given news as real or fake using the trained models.
