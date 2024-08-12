**README**
# Fake News Detection using LSTM + BiLSTM

This project focuses on implementing a model for detecting fake news using Long Short-Term Memory (LSTM) and Bidirectional Long Short-Term Memory (BiLSTM) neural networks. The goal is to classify news articles as either "fake" or "real" based on their content. The project involves preprocessing the text data, building and training LSTM and BiLSTM models, and evaluating their performance.

## Project Overview
This project is designed as an assignment to explore the use of LSTM and BiLSTM models in the context of natural language processing (NLP) for the task of fake news detection. The task is to predict whether a given news article is fake or real by processing the text data and passing it through the neural network models.

## Key Objectives
- **Text Preprocessing:** Clean and preprocess the news articles to prepare them for model input.
- **Model Implementation:** Implement LSTM and BiLSTM models for binary classification.
- **Model Training:** Train the models on a dataset of labeled news articles.
- **Performance Evaluation:** Evaluate the models' performance using metrics such as accuracy, precision, recall, and F1-score.
- **Comparison:** Compare the performance of LSTM and BiLSTM models to understand the advantages and limitations of each.

## Notebook Structure
### Part 1: Data Preprocessing
- **Text Cleaning:** Implement functions to clean the text data, including removing stop words, punctuation, and performing tokenization.
- **Embedding:** Convert the text data into word embeddings suitable for input to LSTM/BiLSTM models.

### Part 2: Model Building
- **LSTM Model:** Build an LSTM model to process the sequential text data.
- **BiLSTM Model:** Build a Bidirectional LSTM model to capture context from both directions of the text sequence.

### Part 3: Model Training and Evaluation
- **Training:** Train the LSTM and BiLSTM models on the preprocessed dataset.
- **Evaluation:** Evaluate the trained models on a test set using various performance metrics.

### Part 4: Model Comparison and Conclusion
- **Comparison:** Compare the results of the LSTM and BiLSTM models.
- **Conclusion:** Discuss the findings and potential improvements.

## Getting Started
### Prerequisites
To run this project, you will need:
- **Python 3.x:** The programming language used for this project.
- **Jupyter Notebook:** To run the .ipynb or .html files.
- **TensorFlow or Keras:** For building and training the neural network models.
  
### Installation
Clone the Repository:
```bash
git clone https://github.com/MatasT-uni/NLP-Fake-News-Detection-using-LSTM-BiLSTM-by-Python
cd NLP-Fake-News-Detection-using-LSTM-BiLSTM-by-Python
```

Install Required Packages:
```bash
pip install numpy pandas tensorflow keras nltk
```

Run the Notebook:
- Open the Jupyter Notebook or HTML file and run all cells to see the implementation of the LSTM and BiLSTM models for fake news detection.

## Usage
- **Run the notebook** to follow the step-by-step implementation and understand how LSTM and BiLSTM models are applied to fake news detection.
- **Use the trained models** to predict whether new articles are fake or real by feeding them into the models.
- **Analyze the performance** of the models using the provided evaluation metrics.

## Results
The project demonstrates the process of building LSTM and BiLSTM models from scratch for the task of fake news detection. It compares the effectiveness of these models and provides insights into their performance, showcasing the strengths and weaknesses of each approach.
