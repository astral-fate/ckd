# model_definition.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

class CKDNutritionModel:
    def __init__(self):
        self.feature_names = ['Age years', 'Weight', 'Height', 'Blood Urea', 'Serum Creatinine', 'GFR']
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        self.nutrition_recommender = NutritionRecommender()

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        X = self._validate_input(X)
        ckd_stages = self.pipeline.predict(X)
        nutrition_recommendations = self.nutrition_recommender.calculate_nutrition_needs(X, ckd_stages)
        return list(zip(ckd_stages, nutrition_recommendations))

    def _validate_input(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        else:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            X = X[self.feature_names]
        return X