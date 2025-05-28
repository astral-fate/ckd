"""
Validation and Comparison Script for CKD Classification Models
Comparing original and enhanced approaches
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.pipeline import Pipeline
import joblib
import os

# Create directories for results
os.makedirs('./validation_results', exist_ok=True)

def generate_original_dataset(n_samples=60000, random_state=42):
    """
    Generate a dataset using the original methodology with limited biomarkers
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the synthetic dataset with original biomarkers
    """
    np.random.seed(random_state)
    
    # Generate demographic data
    ages = np.random.normal(60, 15, n_samples)
    ages = np.clip(ages, 18, 95)
    gender = np.random.choice(['M', 'F'], size=n_samples)
    
    # Initialize arrays for biomarkers
    creatinine = np.zeros(n_samples)
    gfr = np.zeros(n_samples)
    
    # Generate creatinine with natural variations based on age and gender
    for i in range(n_samples):
        # Age effect on biomarkers
        age_factor = (ages[i] - 60) * 0.005  # Slight increase with age
        
        if gender[i] == 'M':
            # Male base distributions
            base_cr = np.random.gamma(shape=2, scale=0.5)  # Natural right-skewed distribution
            creatinine[i] = np.clip(base_cr + age_factor, 0.7, 5.0)
        else:
            # Female base distributions (typically lower)
            base_cr = np.random.gamma(shape=1.8, scale=0.45)
            creatinine[i] = np.clip(base_cr + age_factor, 0.6, 4.5)
    
    # Calculate GFR using CKD-EPI Creatinine equation (2009)
    for i in range(n_samples):
        k = 0.7 if gender[i] == 'F' else 0.9
        a = -0.329 if gender[i] == 'F' else -0.411
        
        # Add natural biological variation
        biological_variation = np.random.normal(1, 0.05)
        
        gfr[i] = (142 * min(creatinine[i]/k, 1)**a * max(creatinine[i]/k, 1)**-1.200 * 
                 0.9938**ages[i] * (1.012 if gender[i] == 'F' else 1)) * biological_variation
    
    # Generate other traditional biomarkers based on GFR values
    blood_urea = 20 * (90/np.maximum(gfr, 1)) * np.random.normal(1, 0.2, n_samples)
    blood_urea = np.clip(blood_urea, 7, 200)
    
    # Hemoglobin with gender-specific ranges and CKD-related anemia
    hemoglobin = np.zeros(n_samples)
    for i in range(n_samples):
        if gender[i] == 'M':
            base_hgb = 15 * (0.7 + 0.3 * (gfr[i]/90))
            hemoglobin[i] = np.clip(base_hgb + np.random.normal(0, 1), 6, 17.5)
        else:
            base_hgb = 13.5 * (0.7 + 0.3 * (gfr[i]/90))
            hemoglobin[i] = np.clip(base_hgb + np.random.normal(0, 1), 6, 15.5)
    
    # Electrolytes with natural relationships to GFR
    potassium = 4.0 + np.maximum(0, (60 - gfr)/60) * np.random.normal(1, 0.1, n_samples)
    potassium = np.clip(potassium, 3.0, 7.0)
    
    sodium = np.where(gfr < 30,
                     np.random.normal(137, 4, n_samples),
                     np.random.normal(140, 2, n_samples))
    sodium = np.clip(sodium, 125, 150)
    
    # Create DataFrame with original biomarkers
    data = pd.DataFrame({
        'Age': np.round(ages, 1),
        'Gender': gender,
        'Creatinine': np.round(creatinine, 2),
        'GFR': np.round(gfr, 1),
        'Blood_Urea': np.round(blood_urea, 1),
        'Hemoglobin': np.round(hemoglobin, 1),
        'Potassium': np.round(potassium, 1),
        'Sodium': np.round(sodium, 1)
    })
    
    # Add CKD stages based on GFR values
    conditions = [
        (data['GFR'] >= 90),  # Stage 1
        (data['GFR'] >= 60) & (data['GFR'] < 90),  # Stage 2
        (data['GFR'] >= 45) & (data['GFR'] < 60),  # Stage 3a
        (data['GFR'] >= 30) & (data['GFR'] < 45),  # Stage 3b
        (data['GFR'] >= 15) & (data['GFR'] < 30),  # Stage 4
        (data['GFR'] < 15)  # Stage 5
    ]
    stages = ['Stage 1', 'Stage 2', 'Stage 3a', 'Stage 3b', 'Stage 4', 'Stage 5']
    data['CKD_Stage'] = np.select(conditions, stages)
    
    return data

def balance_dataset(data, samples_per_class=7500):
    """
    Balance the dataset to have equal representation of all classes
    
    Parameters:
    -----------
    data : pd.DataFrame
        Original dataset
    samples_per_class : int
        Number of samples per class in the balanced dataset
        
    Returns:
    --------
    pd.DataFrame
        Balanced dataset
    """
    from sklearn.utils import resample
    
    stages = data['CKD_Stage'].unique()
    
    balanced_dfs = []
    for stage in stages:
        stage_data = data[data['CKD_Stage'] == stage]
        if len(stage_data) < samples_per_class:
            # Oversample if we have too few samples
            stage_balanced = resample(stage_data, 
                                   replace=True,
                                   n_samples=samples_per_class,
                                   random_state=42)
        else:
            # Undersample if we have too many samples
            stage_balanced = resample(stage_data,
                                   replace=False,
                                   n_samples=samples_per_class,
                                   random_state=42)
        balanced_dfs.append(stage_balanced)
    
    return pd.concat(balanced_dfs, ignore_index=True)

def split_dataset(data, test_size=0.2, validation_size=0.2, random_state=42):
    """
    Split the dataset into training, validation, and test sets
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset to split
    test_size : float
        Proportion of the dataset to include in the test split
    validation_size : float
        Proportion of the training data to include in the validation split
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing X_train, X_val, X_test, y_train, y_val, y_test
    """
    from sklearn.model_selection import train_test_split
    
    # First split into train+val and test
    X = data.drop(['CKD_Stage'], axis=1)
    y = data['CKD_Stage']
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Then split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=validation_size/(1-test_size),  # Adjust validation size
        random_state=random_state, 
        stratify=y_train_val
    )
    
    return {
        'X_train': X_train, 
        'X_val': X_val, 
        'X_test': X_test, 
        'y_train': y_train, 
        'y_val': y_val, 
        'y_test': y_test
    }

def train_and_evaluate_models(data, model_type='xgboost'):
    """
    Train and evaluate models on the dataset
    
    Parameters:
    -----------
    data : dict
        Dictionary containing X_train, X_val, X_test, y_train, y_val, y_test
    model_type : str
        Type of model to train ('xgboost', 'random_forest', or 'gradient_boosting')
        
    Returns:
    --------
    dict
        Dictionary containing model and performance metrics
    """
    # Extract data
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Handle categorical features
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_val = pd.get_dummies(X_val, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    
    # Ensure all datasets have the same columns
    all_columns = X_train.columns
    X_val = X_val.reindex(columns=all_columns, fill_value=0)
    X_test = X_test.reindex(columns=all_columns, fill_value=0)
    
    # Create model
    if model_type == 'xgboost':
        model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=2,
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            random_state=42
        )
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=200, 
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_preds = pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_preds)
    val_report = classification_report(y_val, val_preds, output_dict=True)
    
    # Evaluate on test set
    test_preds = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_preds)
    test_report = classification_report(y_test, test_preds, output_dict=True)
    
    # Generate confusion matrices
    val_cm = confusion_matrix(y_val, val_preds)
    test_cm = confusion_matrix(y_test, test_preds)
    
    # Store results
    results = {
        'model': pipeline,
        'val_accuracy': val_accuracy,
        'val_report': val_report,
        'val_cm': val_cm,
        'test_accuracy': test_accuracy,
        'test_report': test_report,
        'test_cm': test_cm
    }
    
    return results

def compare_models(original_results, enhanced_results, output_dir='./validation_results'):
    """
    Compare the performance of original and enhanced models
    
    Parameters:
    -----------
    original_results : dict
        Results from the original model
    enhanced_results : dict
        Results from the enhanced model
    output_dir : str
        Directory to save comparison results
    """
    # Create comparison table
    comparison = pd.DataFrame({
        'Metric': ['Test Accuracy', 'Macro F1-Score', 'Weighted F1-Score'],
        'Original Model': [
            original_results['test_accuracy'],
            original_results['test_report']['macro avg']['f1-score'],
            original_results['test_report']['weighted avg']['f1-score']
        ],
        'Enhanced Model': [
            enhanced_results['test_accuracy'],
            enhanced_results['test_report']['macro avg']['f1-score'],
            enhanced_results['test_report']['weighted avg']['f1-score']
        ],
        'Improvement': [
            enhanced_results['test_accuracy'] - original_results['test_accuracy'],
            enhanced_results['test_report']['macro avg']['f1-score'] - original_results['test_report']['macro avg']['f1-score'],
            enhanced_results['test_report']['weighted avg']['f1-score'] - original_results['test_report']['weighted avg']['f1-score']
        ]
    })
    
    # Save comparison table
    comparison.to_csv(f'{output_dir}/model_comparison.csv', index=False)
    
    # Create stage-specific comparison
    stages = list(original_results['test_report'].keys())
    stages = [s for s in stages if s not in ['accuracy', 'macro avg', 'weighted avg']]
    
    stage_comparison = []
    for stage in stages:
        if stage in enhanced_results['test_report']:
            stage_comparison.append({
                'Stage': stage,
                'Original Precision': original_results['test_report'][stage]['precision'],
                'Enhanced Precision': enhanced_results['test_report'][stage]['precision'],
                'Precision Improvement': enhanced_results['test_report'][stage]['precision'] - original_results['test_report'][stage]['precision'],
                'Original Recall': original_results['test_report'][stage]['recall'],
                'Enhanced Recall': enhanced_results['test_report'][stage]['recall'],
                'Recall Improvement': enhanced_results['test_report'][stage]['recall'] - original_results['test_report'][stage]['recall'],
                'Original F1': original_results['test_report'][stage]['f1-score'],
                'Enhanced F1': enhanced_results['test_report'][stage]['f1-score'],
                'F1 Improvement': enhanced_results['test_report'][stage]['f1-score'] - original_results['test_report'][stage]['f1-score']
            })
    
    stage_comparison_df = pd.DataFrame(stage_comparison)
    stage_comparison_df.to_csv(f'{output_dir}/stage_comparison.csv', index=False)
    
    # Create visualizations
    
    # 1. Overall performance comparison
    plt.figure(figsize=(10, 6))
    metrics = ['Test Accuracy', 'Macro F1-Score', 'Weighted F1-Score']
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, comparison['Original Model'], width, label='Original Model')
    plt.bar(x + width/2, comparison['Enhanced Model'], width, label='Enhanced Model')
    
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Performance Comparison: Original vs. Enhanced Models')
    plt.xticks(x, metrics)
    plt.ylim(0.9, 1.0)  # Adjust as needed to highlight differences
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/overall_performance_comparison.png', dpi=300)
    plt.close()
    
    # 2. Stage-specific F1 score comparison
    plt.figure(figsize=(12, 6))
    stages = stage_comparison_df['Stage']
    x = np.arange(len(stages))
    width = 0.35
    
    plt.bar(x - width/2, stage_comparison_df['Original F1'], width, label='Original Model')
    plt.bar(x + width/2, stage_comparison_df['Enhanced F1'], width, label='Enhanced Model')
    
    plt.xlabel('CKD Stage')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Comparison by CKD Stage')
    plt.xticks(x, stages, rotation=45)
    plt.ylim(0.9, 1.0)  # Adjust as needed to highlight differences
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/stage_f1_comparison.png', dpi=300)
    plt.close()
    
    # 3. Confusion matrix comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Original model confusion matrix
    sns.heatmap(original_results['test_cm'], annot=True, fmt='d', cmap='Blues', 
               xticklabels=stages, yticklabels=stages, ax=axes[0])
    axes[0].set_title('Original Model Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Enhanced model confusion matrix
    sns.heatmap(enhanced_results['test_cm'], annot=True, fmt='d', cmap='Blues', 
               xticklabels=stages, yticklabels=stages, ax=axes[1])
    axes[1].set_title('Enhanced Model Confusion Matrix')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix_comparison.png', dpi=300)
    plt.close()
    
    # Print summary
    print("\nModel Comparison Summary:")
    print(comparison)
    print("\nStage-specific Comparison:")
    print(stage_comparison_df)
    
    return comparison, stage_comparison_df

def main():
    """
    Main function to validate and compare original and enhanced approaches
    """
    print("Generating datasets for validation...")
    
    # Generate original dataset with limited biomarkers
    original_data = generate_original_dataset(n_samples=100000)
    original_balanced = balance_dataset(original_data, samples_per_class=7500)
    
    # Load enhanced dataset (or generate if not available)
    try:
        enhanced_data = pd.read_csv('ckd_enhanced_dataset.csv')
    except FileNotFoundError:
        print("Enhanced dataset not found. Please run the improved_ckd_dataset_generator.py script first.")
        return
    
    print("\nSplitting datasets...")
    original_splits = split_dataset(original_balanced)
    
    # For enhanced data, we need to use the simplified stages for fair comparison
    enhanced_for_comparison = enhanced_data.copy()
    enhanced_for_comparison['CKD_Stage'] = enhanced_for_comparison['CKD_Stage_Simple']
    enhanced_splits = split_dataset(enhanced_for_comparison)
    
    print("\nTraining and evaluating models...")
    # Train XGBoost models on both datasets with identical hyperparameters
    original_results = train_and_evaluate_models(original_splits, model_type='xgboost')
    enhanced_results = train_and_evaluate_models(enhanced_splits, model_type='xgboost')
    
    print("\nComparing model performance...")
    comparison, stage_comparison = compare_models(original_results, enhanced_results)
    
    print("\nValidation and comparison completed successfully!")
    print(f"Results saved in ./validation_results/")

if __name__ == "__main__":
    main()
