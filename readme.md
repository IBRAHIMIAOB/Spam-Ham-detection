# Naive Bayes Text Classification Project

## Project Description

This project implements a Naive Bayes text classification system capable of handling binary or multi-class classification tasks. The system is built with Python and includes:

- **NaiveBayes.py**: The main implementation of the Naive Bayes classifier.
- **main.py**: A script to train, test, and evaluate the model.
- **presentation.pptx**: A presentation describing the project in detail.

---

## Files in the Project

### 1. **NaiveBayes.py**

This file contains the implementation of the Naive Bayes classifier with the following key features:

- **Initialization**: Defines categories and vocabulary.
- **Training**:
  - `fit`: Train the model with a single text and category.
  - `fit_all`: Train the model with multiple texts and corresponding categories.
- **Testing**:
  - `test`: Classify a given text based on the highest category score.
- **Evaluation**:
  - `evaluate_predictions`: Calculate accuracy over multiple texts and labels.

### 2. **main.py**

This file demonstrates how to use the Naive Bayes classifier:

- **Training**:
  - `nb_train(x, y)`: Trains the model using training data.
- **Testing**:
  - `nb_test(docs, model, use_log, smoothing)`: Tests the model on new documents and returns predictions.
- **Evaluation**:
  - `f_score(y_true, y_pred)`: Calculates the F1 score for model evaluation.

### 3. **presentation.pptx**

A presentation file summarizing the project, including the implementation details and results.

---

## How to Use

### Step 1: Training the Model

Train the Naive Bayes model with your dataset:

```python
from NaiveBayes import NaiveBayes

# Example usage
x_train = ["text1", "text2", "text3"]
y_train = ["Category1", "Category2", "Category1"]

model = nb_train(x_train, y_train)
```

### Step 2: Testing the Model

Test the model on new data:

```python
x_test = ["new_text1", "new_text2"]
y_test = ["Category1", "Category2"]

predictions = nb_test(x_test, model, use_log=True, smoothing=True)
```

### Step 3: Evaluate Model Performance

Calculate the F1 score:

```python
f1 = f_score(y_test, predictions)
print(f"F1 Score: {f1}")
```

---

## Features

- Supports both logarithmic probabilities and standard probabilities for classification.
- Implements smoothing to handle unseen words.
- Calculates precision, recall, and F1 score for performance evaluation.

---

## Example Results

The performance of the model is evaluated using various combinations of logarithmic probabilities and smoothing:

- **Logarithmic probabilities with smoothing**
- **Logarithmic probabilities without smoothing**
- **Standard probabilities with smoothing**
- **Standard probabilities without smoothing**

Example results are printed in the following format:

```plaintext
smoothing = True , log = True
F1 Score: 0.85
smoothing = True , log = False
F1 Score: 0.80
```

---

## Prerequisites

- Python 3.6+
- Basic understanding of Naive Bayes classification.

---

## References

For further details, refer to the accompanying `presentation.pptx` file.

