# Chronic Kidney Disease (CKD) Prediction and Nutrition Recommender

## About the Project 🩺

This project is a web application designed to predict the stage of Chronic Kidney Disease (CKD) and provide personalized nutrition recommendations based on the user's health data. The application uses a machine learning model to classify the CKD stage and a rule-based system to generate a diet plan. This tool can be a helpful resource for individuals and healthcare professionals to manage CKD.

---

## Features ✨

* **CKD Stage Prediction**: Predicts the stage of CKD (Stage 1 to 5) based on user-provided health metrics.
* **Nutrition Recommendation**: Generates a personalized diet plan with recommended daily intake of calories, protein, carbohydrates, fat, sodium, potassium, and phosphorus based on the predicted CKD stage and user's age, weight, and height.
* **User-friendly Web Interface**: An interactive front-end that allows users to input their data and view the results.
* **Data Visualization**: The project includes comprehensive data visualizations to understand the dataset and model performance.

---

## Dataset 📊

The model is trained on a balanced dataset of 60,000 samples, with 10,000 samples for each of the six CKD stages. The dataset was created by generating synthetic data and then balancing it to ensure equal representation of each stage.

The features used for training the model are:

* Age
* Creatinine
* Glomerular Filtration Rate (GFR)
* Blood Urea
* Hemoglobin
* Potassium
* Sodium
* Gender

---

## Model 🤖

The prediction model is a machine learning pipeline that includes a scaler and a classifier. The `ckd.ipynb` notebook shows the training and evaluation of several models, including **Logistic Regression**, **Support Vector Machine (SVM)**, **Decision Tree**, **Random Forest**, and **K-Nearest Neighbors (KNN)**. The final model used in the application is a pre-trained pipeline loaded from `ckd_pipeline.pkl`. The `improved_ckd_classification.py` file suggests further improvements to the classification model.

The application uses a `NutritionRecommender` class to calculate the nutritional needs based on the predicted CKD stage and user's BMI.

---

## Getting Started 🚀

To get a local copy up and running, follow these simple steps.

### Prerequisites

* Python 3.x
* Flask
* Pandas
* NumPy
* Scikit-learn

### Installation

1.  Clone the repo
    ```sh
    git clone [https://github.com/ckd/ckd.git](https://github.com/ckd/ckd.git)
    ```
2.  Install Python packages
    ```sh
    pip install Flask pandas numpy scikit-learn
    ```

### Usage

1.  Run the Flask application:
    ```sh
    python app.py
    ```
2.  Open your browser and go to `http://127.0.0.1:5000/`

---

## Built With 🛠️

* [Flask](https://flask.palletsprojects.com/) - The web framework used
* [Pandas](https://pandas.pydata.org/) - For data manipulation
* [Scikit-learn](https://scikit-learn.org/) - For building the machine learning model
* [Jupyter Notebook](https://jupyter.org/) - For data analysis and model training
