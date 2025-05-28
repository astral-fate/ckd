"""
Enhanced Machine Learning Models for Multi-Class CKD Classification
Implementing advanced methodologies for improved classification performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel, RFE
import xgboost as xgb
from imblearn.pipeline import Pipeline
import joblib
import os

def load_data(train_path, val_path, test_path, target_column='CKD_Stage', simple_stages=False):
    """
    Load and prepare the train, validation, and test datasets
    
    Parameters:
    -----------
    train_path : str
        Path to the training dataset CSV
    val_path : str
        Path to the validation dataset CSV
    test_path : str
        Path to the test dataset CSV
    target_column : str
        Name of the target column
    simple_stages : bool
        Whether to use simplified CKD stages
        
    Returns:
    --------
    dict
        Dictionary containing X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Load datasets
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)
    
    # Use simple stages if specified
    if simple_stages and 'CKD_Stage_Simple' in train_data.columns:
        target_column = 'CKD_Stage_Simple'
    
    # Separate features and target
    feature_columns = [col for col in train_data.columns if col not in 
                      ['CKD_Stage', 'CKD_Stage_Simple', 'Albuminuria_Category', 'KDIGO_Risk_Category']]
    
    X_train = train_data[feature_columns]
    y_train = train_data[target_column]
    
    X_val = val_data[feature_columns]
    y_val = val_data[target_column]
    
    X_test = test_data[feature_columns]
    y_test = test_data[target_column]
    
    # Handle categorical features
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_val = pd.get_dummies(X_val, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    
    # Ensure all datasets have the same columns
    all_columns = X_train.columns
    X_val = X_val.reindex(columns=all_columns, fill_value=0)
    X_test = X_test.reindex(columns=all_columns, fill_value=0)
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }

def feature_selection(X_train, y_train, method='rfe', n_features=15):
    """
    Perform feature selection to identify the most important features
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    method : str
        Feature selection method ('rfe', 'model_based', or 'combined')
    n_features : int
        Number of features to select
        
    Returns:
    --------
    list
        List of selected feature names
    """
    if method == 'rfe':
        # Recursive Feature Elimination
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        selector = RFE(estimator, n_features_to_select=n_features, step=1)
        selector.fit(X_train, y_train)
        selected_features = X_train.columns[selector.support_]
        
    elif method == 'model_based':
        # Model-based feature selection
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        estimator.fit(X_train, y_train)
        selector = SelectFromModel(estimator, threshold='median', max_features=n_features)
        selector.fit(X_train, y_train)
        selected_features = X_train.columns[selector.get_support()]
        
    elif method == 'combined':
        # Combine both methods
        # First use RFE to get a larger set of features
        rfe_estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        rfe_selector = RFE(rfe_estimator, n_features_to_select=n_features*2, step=1)
        rfe_selector.fit(X_train, y_train)
        rfe_features = X_train.columns[rfe_selector.support_]
        
        # Then use model-based selection on the RFE-selected features
        X_train_rfe = X_train[rfe_features]
        model_estimator = GradientBoostingClassifier(random_state=42)
        model_estimator.fit(X_train_rfe, y_train)
        model_selector = SelectFromModel(model_estimator, threshold='median', max_features=n_features)
        model_selector.fit(X_train_rfe, y_train)
        selected_features = rfe_features[model_selector.get_support()]
    
    else:
        raise ValueError("Method must be one of 'rfe', 'model_based', or 'combined'")
    
    return list(selected_features)

def build_models():
    """
    Build a dictionary of machine learning models for CKD classification
    
    Returns:
    --------
    dict
        Dictionary of model names and their corresponding estimators
    """
    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0, 
            solver='lbfgs', 
            max_iter=1000, 
            multi_class='multinomial',
            random_state=42
        ),
        
        'SVM': SVC(
            C=10, 
            kernel='rbf', 
            gamma='scale', 
            probability=True,
            random_state=42
        ),
        
        'Random Forest': RandomForestClassifier(
            n_estimators=200, 
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42
        ),
        
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42
        ),
        
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=2,
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            random_state=42
        ),
        
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        ),
        
        'KNN': KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            algorithm='auto',
            leaf_size=30,
            p=2  # Euclidean distance
        )
    }
    
    # Add ensemble model
    models['Voting Ensemble'] = VotingClassifier(
        estimators=[
            ('rf', models['Random Forest']),
            ('gb', models['Gradient Boosting']),
            ('xgb', models['XGBoost'])
        ],
        voting='soft'
    )
    
    return models

def tune_hyperparameters(model_name, X_train, y_train, X_val, y_val):
    """
    Tune hyperparameters for a specific model using grid search
    
    Parameters:
    -----------
    model_name : str
        Name of the model to tune
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation target
        
    Returns:
    --------
    object
        Best estimator from grid search
    """
    if model_name == 'Logistic Regression':
        param_grid = {
            'C': [0.1, 0.5, 1.0, 5.0, 10.0],
            'solver': ['lbfgs', 'newton-cg', 'sag'],
            'max_iter': [1000, 2000]
        }
        estimator = LogisticRegression(multi_class='multinomial', random_state=42)
        
    elif model_name == 'SVM':
        param_grid = {
            'C': [1, 5, 10, 50],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'poly']
        }
        estimator = SVC(probability=True, random_state=42)
        
    elif model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        estimator = RandomForestClassifier(random_state=42)
        
    elif model_name == 'Gradient Boosting':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'subsample': [0.8, 0.9, 1.0]
        }
        estimator = GradientBoostingClassifier(random_state=42)
        
    elif model_name == 'XGBoost':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 2, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        estimator = xgb.XGBClassifier(objective='multi:softprob', random_state=42)
        
    elif model_name == 'Neural Network':
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'batch_size': ['auto', 32, 64]
        }
        estimator = MLPClassifier(max_iter=500, random_state=42)
        
    elif model_name == 'KNN':
        param_grid = {
            'n_neighbors': [5, 7, 9, 11, 13],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]  # Manhattan or Euclidean distance
        }
        estimator = KNeighborsClassifier()
        
    else:
        raise ValueError(f"Model {model_name} not supported for hyperparameter tuning")
    
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', estimator)
    ])
    
    # Use stratified k-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid={"model__" + key: val for key, val in param_grid.items()},
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit on training data
    grid_search.fit(X_train, y_train)
    
    # Get best estimator
    best_pipeline = grid_search.best_estimator_
    
    # Evaluate on validation set
    val_accuracy = best_pipeline.score(X_val, y_val)
    print(f"Best {model_name} validation accuracy: {val_accuracy:.4f}")
    print(f"Best parameters: {grid_search.best_params_}")
    
    return best_pipeline

def train_and_evaluate_models(data, selected_features=None, output_dir='./model_results'):
    """
    Train and evaluate multiple machine learning models
    
    Parameters:
    -----------
    data : dict
        Dictionary containing X_train, X_val, X_test, y_train, y_val, y_test
    selected_features : list or None
        List of selected feature names. If None, all features are used.
    output_dir : str
        Directory to save model results
        
    Returns:
    --------
    dict
        Dictionary of trained models and their performance metrics
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract data
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Use selected features if provided
    if selected_features is not None:
        X_train = X_train[selected_features]
        X_val = X_val[selected_features]
        X_test = X_test[selected_features]
    
    # Build models
    models = build_models()
    
    # Dictionary to store results
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
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
        results[name] = {
            'model': pipeline,
            'val_accuracy': val_accuracy,
            'val_report': val_report,
            'val_cm': val_cm,
            'test_accuracy': test_accuracy,
            'test_report': test_report,
            'test_cm': test_cm
        }
        
        # Print results
        print(f"{name} - Validation Accuracy: {val_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        # Save model
        joblib.dump(pipeline, f"{output_dir}/{name.replace(' ', '_').lower()}_model.pkl")
        
        # Plot confusion matrix for validation set
        plt.figure(figsize=(10, 8))
        sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=np.unique(y_val), 
                   yticklabels=np.unique(y_val))
        plt.title(f'{name} - Validation Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{name.replace(' ', '_').lower()}_val_cm.png", dpi=300)
        plt.close()
        
        # Plot confusion matrix for test set
        plt.figure(figsize=(10, 8))
        sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=np.unique(y_test), 
                   yticklabels=np.unique(y_test))
        plt.title(f'{name} - Test Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{name.replace(' ', '_').lower()}_test_cm.png", dpi=300)
        plt.close()
    
    # Compare model performances
    val_accuracies = [results[name]['val_accuracy'] for name in models.keys()]
    test_accuracies = [results[name]['test_accuracy'] for name in models.keys()]
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, val_accuracies, width, label='Validation')
    plt.bar(x + width/2, test_accuracies, width, label='Test')
    
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models.keys(), rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png", dpi=300)
    plt.close()
    
    # Generate summary report
    summary = pd.DataFrame({
        'Model': list(models.keys()),
        'Validation Accuracy': val_accuracies,
        'Test Accuracy': test_accuracies
    })
    
    summary.to_csv(f"{output_dir}/model_summary.csv", index=False)
    
    return results

def analyze_feature_importance(models, feature_names, output_dir='./feature_importance'):
    """
    Analyze and visualize feature importance for tree-based models
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    feature_names : list
        List of feature names
    output_dir : str
        Directory to save feature importance plots
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Models that support feature importance
    tree_based_models = ['Random Forest', 'Gradient Boosting', 'XGBoost']
    
    for model_name in tree_based_models:
        if model_name in models:
            # Extract the actual model from the pipeline
            pipeline = models[model_name]['model']
            model = pipeline.named_steps['model']
            
            # Get feature importance
            if model_name == 'XGBoost':
                importances = model.feature_importances_
            else:
                importances = model.feature_importances_
            
            # Create DataFrame for visualization
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            })
            
            # Sort by importance
            feature_importance = feature_importance.sort_values('Importance', ascending=False)
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
            plt.title(f'Top 20 Feature Importance - {model_name}')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{model_name.replace(' ', '_').lower()}_feature_importance.png", dpi=300)
            plt.close()
            
            # Save feature importance to CSV
            feature_importance.to_csv(f"{output_dir}/{model_name.replace(' ', '_').lower()}_feature_importance.csv", index=False)

def main():
    """
    Main function to run the enhanced CKD classification pipeline
    """
    print("Loading datasets...")
    data = load_data(
        train_path='ckd_train_set.csv',
        val_path='ckd_validation_set.csv',
        test_path='ckd_test_set.csv',
        simple_stages=False  # Use detailed stages
    )
    
    print("\nPerforming feature selection...")
    selected_features = feature_selection(
        data['X_train'], 
        data['y_train'], 
        method='combined', 
        n_features=20
    )
    
    print(f"Selected features: {selected_features}")
    
    print("\nTraining and evaluating models with selected features...")
    results = train_and_evaluate_models(
        data, 
        selected_features=selected_features,
        output_dir='./model_results'
    )
    
    print("\nAnalyzing feature importance...")
    analyze_feature_importance(
        results, 
        selected_features,
        output_dir='./feature_importance'
    )
    
    print("\nHyperparameter tuning for best model...")
    # Find the best performing model on validation set
    best_model_name = max(results.items(), key=lambda x: x[1]['val_accuracy'])[0]
    print(f"Best model: {best_model_name}")
    
    # Tune hyperparameters for the best model
    best_model = tune_hyperparameters(
        best_model_name,
        data['X_train'][selected_features],
        data['y_train'],
        data['X_val'][selected_features],
        data['y_val']
    )
    
    # Save the tuned model
    joblib.dump(best_model, f"./model_results/{best_model_name.replace(' ', '_').lower()}_tuned_model.pkl")
    
    print("\nCKD classification pipeline completed successfully!")

if __name__ == "__main__":
    main()
