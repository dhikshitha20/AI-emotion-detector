# AI Emotion Detector

An advanced AI-powered emotion detection application built using Streamlit and Hugging Face Transformers. The system analyzes text input or uploaded .txt files to identify emotions with confidence scores, supported by interactive visualizations and PDF report generation.

## Overview
AI Emotion Detector classifies text into seven emotional categories using state-of-the-art NLP. The application provides real-time predictions, visual insights, and downloadable reports.

## Features
- Emotion detection from typed text
- Support for .txt file uploads
- Classification into 7 emotions (Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral)
- Confidence score for each prediction
- Interactive bar charts and emotion trend analysis
- PDF report generation and download
- Clean, user-friendly web interface

## Tech Stack
- Python
- Streamlit
- Hugging Face Transformers
- Plotly
- Pandas
- fpdf2

## How to Run
```bash
pip install streamlit transformers torch plotly pandas fpdf2
streamlit run app.py
```
