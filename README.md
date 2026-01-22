# ML-Practice
Practicing machine learning models on multiple datasets using scikit learn
# Machine Learning Practice – Iris , Housing Prediction & Adult Datasets

This repository contains my practice code for training and testing machine learning models using **scikit-learn**.  
I worked with two well-known datasets — **Iris** and **Adult (Census Income)** — to understand the full ML workflow.

---

## Datasets Used

### 1. Iris Dataset
- Type: Classification
- Features:
  - Sepal length
  - Sepal width
  - Petal length
  - Petal width
- Target:
  - Iris-setosa
  - Iris-versicolor
  - Iris-virginica

The Iris dataset contains only numerical features, so it can be trained directly without encoding.

---

### 2. Adult (Census Income) Dataset
- Source: UCI Machine Learning Repository
- Type: Classification
- Task: Predict whether income is `<=50K` or `>50K`
- Contains both:
  - Numerical features (age, hours-per-week, etc.)
  - Categorical features (workclass, education, occupation, etc.)

Categorical features are converted into numerical form using **One-Hot Encoding** before model training.

---

## Models Used

- **RandomForestClassifier**
- (Gradient Boosting imported for experimentation)

Random Forest was used because it handles non-linear relationships well and works effectively on both datasets.

---

## Workflow Followed

1. Load data using pandas
2. Split features (`X`) and target (`y`)
3. Encode categorical data using `OneHotEncoder` (Adult dataset only)
4. Train the model using `RandomForestClassifier`
5. Make predictions on:
   - Single input values
   - Test dataset
6. Evaluate predictions (basic accuracy)

---

## Project Structure


