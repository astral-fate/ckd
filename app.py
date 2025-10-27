from flask import Flask, request, render_template, url_for
import pickle
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

app = Flask(__name__)

# Load the pipeline and scaler
with open('ckd_pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)
with open('ckd_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the NutritionRecommender class
class NutritionRecommender(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def calculate_bmi(self, weight, height):
        return weight / ((height / 100) ** 2)

    def calculate_nutrition_needs(self, X, ckd_stages):
        nutrition_list = []
        for i, stage in enumerate(ckd_stages):
            age = X['Age years'].iloc[i]
            weight = X['Weight'].iloc[i]
            height = X['Height'].iloc[i]
            gfr = X['GFR'].iloc[i]
            
            bmi = self.calculate_bmi(weight, height)
            
            protein_per_kg = max(0.6, min(0.8, 1.2 - 0.1 * stage))
            calories_per_kg = 35 - (stage * 2)
            
            if bmi < 18.5:
                calories_per_kg += 5
            elif bmi >= 30:
                calories_per_kg -= 5
            
            if age > 60:
                calories_per_kg -= 2
            
            total_calories = weight * calories_per_kg
            protein_grams = weight * protein_per_kg
            carbs_grams = (total_calories * 0.55) / 4
            fat_grams = (total_calories * 0.30) / 9
            
            if gfr < 60:
                protein_grams *= 0.9
            
            sodium_mg = 2000 - (stage * 200)
            potassium_mg = max(2000, 4000 - (stage * 400))
            phosphorus_mg = max(800, 1200 - (stage * 100))
            
            nutrition = {
                "calories": round(total_calories),
                "protein": round(protein_grams),
                "carbs": round(carbs_grams),
                "fat": round(fat_grams),
                "sodium": round(sodium_mg),
                "potassium": round(potassium_mg),
                "phosphorus": round(phosphorus_mg)
            }
            nutrition_list.append(nutrition)
        
        return nutrition_list

nutrition_recommender = NutritionRecommender()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        age = float(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        blood_urea = float(request.form['blood_urea'])
        serum_creatinine = float(request.form['serum_creatinine'])
        gfr = float(request.form['gfr'])

        # Create DataFrame for prediction
        input_data = pd.DataFrame({
            'Age years': [age],
            'Weight': [weight],
            'Height': [height],
            'Blood Urea': [blood_urea],
            'Serum Creatinine': [serum_creatinine],
            'GFR': [gfr]
        })

        # Make prediction
        ckd_stage = pipeline.predict(input_data)[0]
        
        # Generate nutrition recommendations
        nutrition = nutrition_recommender.calculate_nutrition_needs(input_data, [ckd_stage])[0]
        
        return render_template('results.html', ckd_stage=ckd_stage, nutrition=nutrition)
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

@app.route('/profile')
def profile():
    # You can add logic here to fetch user profile data if needed
    return render_template('profile.html')

@app.route('/dietplan')
def dietplan():
    # You can add logic here to generate or fetch diet plan data if needed
    return render_template('dietplan.html')

@app.route('/dashboard')
def dashboard():
    # You can add logic here to generate or fetch dashboard data if needed
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)