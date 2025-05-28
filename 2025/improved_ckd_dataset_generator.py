"""
Enhanced Synthetic Dataset Generator for Multi-Class CKD Classification
Incorporating additional biomarkers and improved physiological relationships
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

def generate_enhanced_ckd_dataset(n_samples=60000, random_state=42):
    """
    Generate an enhanced synthetic dataset for CKD classification with additional biomarkers
    and improved physiological relationships based on recent literature.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the synthetic dataset
    """
    np.random.seed(random_state)
    
    # Generate demographic data with natural age distribution
    ages = np.random.normal(60, 15, n_samples)
    ages = np.clip(ages, 18, 95)
    gender = np.random.choice(['M', 'F'], size=n_samples)
    
    # Initialize arrays for all biomarkers
    creatinine = np.zeros(n_samples)
    cystatin_c = np.zeros(n_samples)
    gfr = np.zeros(n_samples)
    gfr_cystatin = np.zeros(n_samples)
    gfr_combined = np.zeros(n_samples)
    
    # Generate creatinine and cystatin C with natural variations based on age and gender
    for i in range(n_samples):
        # Age effect on biomarkers
        age_factor = (ages[i] - 60) * 0.005  # Slight increase with age
        
        if gender[i] == 'M':
            # Male base distributions
            base_cr = np.random.gamma(shape=2, scale=0.5)  # Natural right-skewed distribution
            creatinine[i] = np.clip(base_cr + age_factor, 0.7, 5.0)
            
            # Cystatin C - less affected by muscle mass but still has natural variation
            base_cys = np.random.gamma(shape=2.2, scale=0.45)
            cystatin_c[i] = np.clip(base_cys + age_factor * 0.7, 0.6, 4.5)
        else:
            # Female base distributions (typically lower)
            base_cr = np.random.gamma(shape=1.8, scale=0.45)
            creatinine[i] = np.clip(base_cr + age_factor, 0.6, 4.5)
            
            # Cystatin C - less gender difference than creatinine
            base_cys = np.random.gamma(shape=2.1, scale=0.43)
            cystatin_c[i] = np.clip(base_cys + age_factor * 0.7, 0.6, 4.3)
    
    # Calculate GFR using different equations
    for i in range(n_samples):
        # 1. CKD-EPI Creatinine equation (2009)
        k = 0.7 if gender[i] == 'F' else 0.9
        a = -0.329 if gender[i] == 'F' else -0.411
        
        # Add natural biological variation
        biological_variation = np.random.normal(1, 0.05)
        
        gfr[i] = (142 * min(creatinine[i]/k, 1)**a * max(creatinine[i]/k, 1)**-1.200 * 
                 0.9938**ages[i] * (1.012 if gender[i] == 'F' else 1)) * biological_variation
        
        # 2. CKD-EPI Cystatin C equation (2012)
        gfr_cystatin[i] = 133 * min(cystatin_c[i]/0.8, 1)**-0.499 * max(cystatin_c[i]/0.8, 1)**-1.328 * 0.996**ages[i] * (0.932 if gender[i] == 'F' else 1)
        
        # 3. CKD-EPI Creatinine-Cystatin C equation (2012)
        gfr_combined[i] = 135 * min(creatinine[i]/k, 1)**a * max(creatinine[i]/k, 1)**-0.601 * \
                          min(cystatin_c[i]/0.8, 1)**-0.375 * max(cystatin_c[i]/0.8, 1)**-0.711 * \
                          0.995**ages[i] * (0.969 if gender[i] == 'F' else 1)
    
    # Generate other traditional biomarkers based on GFR values
    blood_urea = 20 * (90/np.maximum(gfr_combined, 1)) * np.random.normal(1, 0.2, n_samples)
    blood_urea = np.clip(blood_urea, 7, 200)
    
    # Hemoglobin with gender-specific ranges and CKD-related anemia
    hemoglobin = np.zeros(n_samples)
    for i in range(n_samples):
        if gender[i] == 'M':
            base_hgb = 15 * (0.7 + 0.3 * (gfr_combined[i]/90))
            hemoglobin[i] = np.clip(base_hgb + np.random.normal(0, 1), 6, 17.5)
        else:
            base_hgb = 13.5 * (0.7 + 0.3 * (gfr_combined[i]/90))
            hemoglobin[i] = np.clip(base_hgb + np.random.normal(0, 1), 6, 15.5)
    
    # Electrolytes with natural relationships to GFR
    potassium = 4.0 + np.maximum(0, (60 - gfr_combined)/60) * np.random.normal(1, 0.1, n_samples)
    potassium = np.clip(potassium, 3.0, 7.0)
    
    sodium = np.where(gfr_combined < 30,
                     np.random.normal(137, 4, n_samples),
                     np.random.normal(140, 2, n_samples))
    sodium = np.clip(sodium, 125, 150)
    
    # Calcium and phosphate metabolism (new biomarkers)
    calcium = 9.5 - np.maximum(0, (90 - gfr_combined)/180) * np.random.normal(1, 0.15, n_samples)
    calcium = np.clip(calcium, 7.0, 10.5)
    
    phosphate = 3.5 + np.maximum(0, (90 - gfr_combined)/45) * np.random.normal(1, 0.2, n_samples)
    phosphate = np.clip(phosphate, 2.5, 8.0)
    
    # FGF-23 (rises early in CKD)
    fgf23_base = 50 + np.maximum(0, (120 - gfr_combined)/15)**2 * np.random.normal(1, 0.3, n_samples)
    fgf23 = np.clip(fgf23_base, 40, 1000)
    
    # Generate albumin/creatinine ratio (ACR) - key marker for CKD diagnosis
    # Log-normal distribution to capture the right skew of albuminuria
    log_acr_base = np.random.normal(2, 1, n_samples) + np.maximum(0, (90 - gfr_combined)/18)
    acr = np.exp(log_acr_base)
    acr = np.clip(acr, 1, 3000)
    
    # Kidney injury markers - KIM-1 and NGAL
    # Both increase with decreasing GFR, but with different patterns
    kim1_base = 1 + np.maximum(0, (90 - gfr_combined)/15) * np.random.normal(1, 0.4, n_samples)
    kim1 = np.clip(kim1_base, 0.5, 12)
    
    ngal_base = 20 + np.maximum(0, (90 - gfr_combined)/10)**1.5 * np.random.normal(1, 0.35, n_samples)
    ngal = np.clip(ngal_base, 15, 500)
    
    # Beta-2 microglobulin - filtered by glomeruli, reabsorbed by tubules
    b2m_base = 2 + np.maximum(0, (90 - gfr_combined)/12)**1.2 * np.random.normal(1, 0.25, n_samples)
    b2m = np.clip(b2m_base, 1.5, 25)
    
    # Inflammatory markers - IL-6 and TNF-alpha
    # Both tend to increase with CKD progression
    il6_base = 2 + np.maximum(0, (90 - gfr_combined)/30) * np.random.normal(1, 0.6, n_samples)
    il6 = np.clip(il6_base, 1, 30)
    
    tnf_base = 5 + np.maximum(0, (90 - gfr_combined)/25) * np.random.normal(1, 0.5, n_samples)
    tnf = np.clip(tnf_base, 3, 40)
    
    # Create DataFrame with all biomarkers
    data = pd.DataFrame({
        'Age': np.round(ages, 1),
        'Gender': gender,
        # Traditional biomarkers
        'Creatinine': np.round(creatinine, 2),
        'Cystatin_C': np.round(cystatin_c, 2),
        'GFR': np.round(gfr, 1),
        'GFR_CysC': np.round(gfr_cystatin, 1),
        'GFR_Combined': np.round(gfr_combined, 1),
        'Blood_Urea': np.round(blood_urea, 1),
        'Hemoglobin': np.round(hemoglobin, 1),
        'Potassium': np.round(potassium, 1),
        'Sodium': np.round(sodium, 1),
        # New biomarkers
        'Calcium': np.round(calcium, 1),
        'Phosphate': np.round(phosphate, 1),
        'FGF23': np.round(fgf23, 1),
        'ACR': np.round(acr, 1),
        'KIM1': np.round(kim1, 2),
        'NGAL': np.round(ngal, 1),
        'Beta2_Microglobulin': np.round(b2m, 2),
        'IL6': np.round(il6, 2),
        'TNF_alpha': np.round(tnf, 1)
    })
    
    # Add CKD stages based on combined GFR values (most accurate)
    conditions = [
        (data['GFR_Combined'] >= 90) & (data['ACR'] < 30),  # Stage 1 with normal albuminuria
        (data['GFR_Combined'] >= 90) & (data['ACR'] >= 30),  # Stage 1 with albuminuria
        (data['GFR_Combined'] >= 60) & (data['GFR_Combined'] < 90) & (data['ACR'] < 30),  # Stage 2 with normal albuminuria
        (data['GFR_Combined'] >= 60) & (data['GFR_Combined'] < 90) & (data['ACR'] >= 30),  # Stage 2 with albuminuria
        (data['GFR_Combined'] >= 45) & (data['GFR_Combined'] < 60),  # Stage 3a
        (data['GFR_Combined'] >= 30) & (data['GFR_Combined'] < 45),  # Stage 3b
        (data['GFR_Combined'] >= 15) & (data['GFR_Combined'] < 30),  # Stage 4
        (data['GFR_Combined'] < 15)  # Stage 5
    ]
    stages = ['Stage 1 (Normal)', 'Stage 1 (Albuminuria)', 
              'Stage 2 (Normal)', 'Stage 2 (Albuminuria)', 
              'Stage 3a', 'Stage 3b', 'Stage 4', 'Stage 5']
    data['CKD_Stage'] = np.select(conditions, stages)
    
    # Add simplified CKD stage for backward compatibility
    simplified_conditions = [
        (data['GFR_Combined'] >= 90),  # Stage 1
        (data['GFR_Combined'] >= 60) & (data['GFR_Combined'] < 90),  # Stage 2
        (data['GFR_Combined'] >= 45) & (data['GFR_Combined'] < 60),  # Stage 3a
        (data['GFR_Combined'] >= 30) & (data['GFR_Combined'] < 45),  # Stage 3b
        (data['GFR_Combined'] >= 15) & (data['GFR_Combined'] < 30),  # Stage 4
        (data['GFR_Combined'] < 15)  # Stage 5
    ]
    simplified_stages = ['Stage 1', 'Stage 2', 'Stage 3a', 'Stage 3b', 'Stage 4', 'Stage 5']
    data['CKD_Stage_Simple'] = np.select(simplified_conditions, simplified_stages)
    
    # Add albuminuria categories according to KDIGO guidelines
    albuminuria_conditions = [
        (data['ACR'] < 30),  # A1: Normal to mildly increased
        (data['ACR'] >= 30) & (data['ACR'] < 300),  # A2: Moderately increased
        (data['ACR'] >= 300)  # A3: Severely increased
    ]
    albuminuria_categories = ['A1', 'A2', 'A3']
    data['Albuminuria_Category'] = np.select(albuminuria_conditions, albuminuria_categories)
    
    # Add KDIGO risk categories
    # Create a mapping function for risk categories
    def assign_risk_category(row):
        gfr = row['GFR_Combined']
        alb_cat = row['Albuminuria_Category']
        
        if gfr >= 90:
            if alb_cat == 'A1':
                return 'Low Risk'
            elif alb_cat == 'A2':
                return 'Moderately Increased Risk'
            else:  # A3
                return 'High Risk'
        elif gfr >= 60:
            if alb_cat == 'A1':
                return 'Low Risk'
            elif alb_cat == 'A2':
                return 'Moderately Increased Risk'
            else:  # A3
                return 'High Risk'
        elif gfr >= 45:
            if alb_cat == 'A1':
                return 'Moderately Increased Risk'
            elif alb_cat == 'A2':
                return 'High Risk'
            else:  # A3
                return 'Very High Risk'
        elif gfr >= 30:
            if alb_cat == 'A1':
                return 'High Risk'
            else:  # A2 or A3
                return 'Very High Risk'
        elif gfr >= 15:
            return 'Very High Risk'
        else:
            return 'Very High Risk'
    
    data['KDIGO_Risk_Category'] = data.apply(assign_risk_category, axis=1)
    
    return data

def balance_dataset(data, samples_per_class=7500, use_detailed_stages=True):
    """
    Balance the dataset to have equal representation of all classes
    
    Parameters:
    -----------
    data : pd.DataFrame
        Original dataset
    samples_per_class : int
        Number of samples per class in the balanced dataset
    use_detailed_stages : bool
        Whether to use detailed CKD stages (with albuminuria) or simplified stages
        
    Returns:
    --------
    pd.DataFrame
        Balanced dataset
    """
    stage_column = 'CKD_Stage' if use_detailed_stages else 'CKD_Stage_Simple'
    stages = data[stage_column].unique()
    
    balanced_dfs = []
    for stage in stages:
        stage_data = data[data[stage_column] == stage]
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
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split into train+val and test
    X = data.drop(['CKD_Stage', 'CKD_Stage_Simple', 'Albuminuria_Category', 'KDIGO_Risk_Category'], axis=1)
    y = data['CKD_Stage']
    y_simple = data['CKD_Stage_Simple']
    
    X_train_val, X_test, y_train_val, y_test, y_simple_train_val, y_simple_test = train_test_split(
        X, y, y_simple, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Then split train+val into train and val
    X_train, X_val, y_train, y_val, y_simple_train, y_simple_val = train_test_split(
        X_train_val, y_train_val, y_simple_train_val, 
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
        'y_test': y_test,
        'y_simple_train': y_simple_train,
        'y_simple_val': y_simple_val,
        'y_simple_test': y_simple_test
    }

def create_visualizations(data, output_dir='.'):
    """
    Create comprehensive visualizations of the dataset
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset to visualize
    output_dir : str
        Directory to save visualizations
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set up the style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Distribution of CKD Stages
    plt.figure(figsize=(12, 6))
    sns.countplot(data=data, x='CKD_Stage_Simple', 
                 order=['Stage 1', 'Stage 2', 'Stage 3a', 'Stage 3b', 'Stage 4', 'Stage 5'])
    plt.title('Distribution of CKD Stages')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ckd_stage_distribution.png', dpi=300)
    plt.close()
    
    # 2. Distribution of Albuminuria Categories
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x='Albuminuria_Category', order=['A1', 'A2', 'A3'])
    plt.title('Distribution of Albuminuria Categories')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/albuminuria_distribution.png', dpi=300)
    plt.close()
    
    # 3. Distribution of KDIGO Risk Categories
    plt.figure(figsize=(12, 6))
    risk_order = ['Low Risk', 'Moderately Increased Risk', 'High Risk', 'Very High Risk']
    sns.countplot(data=data, x='KDIGO_Risk_Category', order=risk_order)
    plt.title('Distribution of KDIGO Risk Categories')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/kdigo_risk_distribution.png', dpi=300)
    plt.close()
    
    # 4. Correlation Heatmap
    plt.figure(figsize=(16, 14))
    # Convert gender to numeric
    data_corr = data.copy()
    data_corr['Gender_encoded'] = (data_corr['Gender'] == 'M').astype(int)
    
    # Select numeric columns for correlation
    numeric_cols = data_corr.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = data_corr[numeric_cols].corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(correlation_matrix), k=1)
    
    # Plot heatmap
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                mask=mask,
                square=True,
                fmt='.2f',
                annot_kws={"size": 8})
    plt.title('Correlation Heatmap of Features')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300)
    plt.close()
    
    # 5. GFR vs Key Biomarkers by Gender
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.ravel()
    
    # GFR vs Creatinine
    sns.scatterplot(data=data.sample(2000), x='Creatinine', y='GFR_Combined', hue='Gender', ax=axes[0])
    axes[0].set_title('GFR vs Creatinine by Gender')
    
    # GFR vs Cystatin C
    sns.scatterplot(data=data.sample(2000), x='Cystatin_C', y='GFR_Combined', hue='Gender', ax=axes[1])
    axes[1].set_title('GFR vs Cystatin C by Gender')
    
    # GFR vs ACR
    sns.scatterplot(data=data.sample(2000), x='ACR', y='GFR_Combined', hue='Gender', ax=axes[2])
    axes[2].set_title('GFR vs Albumin-to-Creatinine Ratio by Gender')
    axes[2].set_xscale('log')
    
    # GFR vs Age
    sns.scatterplot(data=data.sample(2000), x='Age', y='GFR_Combined', hue='Gender', ax=axes[3])
    axes[3].set_title('GFR vs Age by Gender')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gfr_key_biomarkers.png', dpi=300)
    plt.close()
    
    # 6. Box plots for CKD stages vs key biomarkers
    biomarkers_to_plot = ['Creatinine', 'Cystatin_C', 'ACR', 'Hemoglobin', 
                          'Phosphate', 'FGF23', 'KIM1', 'NGAL']
    
    for i in range(0, len(biomarkers_to_plot), 2):
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        for j in range(2):
            if i+j < len(biomarkers_to_plot):
                biomarker = biomarkers_to_plot[i+j]
                sns.boxplot(data=data, x='CKD_Stage_Simple', y=biomarker, ax=axes[j])
                axes[j].set_title(f'{biomarker} Levels by CKD Stage')
                axes[j].tick_params(axis='x', rotation=45)
                
                # If plotting ACR, use log scale
                if biomarker == 'ACR':
                    axes[j].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/biomarkers_by_stage_{i}.png', dpi=300)
        plt.close()
    
    # 7. Pairplot of key biomarkers
    key_features = ['GFR_Combined', 'Creatinine', 'Cystatin_C', 'ACR', 'KIM1']
    sns.pairplot(data[key_features].sample(1000), diag_kind='kde')
    plt.suptitle('Pairplot of Key CKD Biomarkers', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/biomarkers_pairplot.png', dpi=300)
    plt.close()
    
    # 8. Comparison of GFR estimation methods
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # GFR Creatinine vs GFR Combined
    sns.scatterplot(data=data.sample(2000), x='GFR', y='GFR_Combined', hue='Gender', ax=axes[0])
    axes[0].plot([0, 150], [0, 150], 'k--')  # Identity line
    axes[0].set_title('GFR Creatinine vs GFR Combined')
    axes[0].set_xlabel('GFR Creatinine (ml/min/1.73m²)')
    axes[0].set_ylabel('GFR Combined (ml/min/1.73m²)')
    
    # GFR Cystatin C vs GFR Combined
    sns.scatterplot(data=data.sample(2000), x='GFR_CysC', y='GFR_Combined', hue='Gender', ax=axes[1])
    axes[1].plot([0, 150], [0, 150], 'k--')  # Identity line
    axes[1].set_title('GFR Cystatin C vs GFR Combined')
    axes[1].set_xlabel('GFR Cystatin C (ml/min/1.73m²)')
    axes[1].set_ylabel('GFR Combined (ml/min/1.73m²)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gfr_methods_comparison.png', dpi=300)
    plt.close()

def main():
    """
    Main function to generate and process the enhanced CKD dataset
    """
    print("Generating enhanced CKD dataset...")
    initial_data = generate_enhanced_ckd_dataset(n_samples=100000)
    
    print("Initial class distribution (simplified stages):")
    print(initial_data['CKD_Stage_Simple'].value_counts())
    
    print("\nInitial class distribution (detailed stages):")
    print(initial_data['CKD_Stage'].value_counts())
    
    # Balance the dataset with 7500 samples per class using detailed stages
    balanced_data = balance_dataset(initial_data, samples_per_class=7500, use_detailed_stages=True)
    
    print("\nBalanced class distribution (detailed stages):")
    print(balanced_data['CKD_Stage'].value_counts())
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(balanced_data, output_dir='./visualizations')
    
    # Split the dataset
    print("\nSplitting dataset into train, validation, and test sets...")
    splits = split_dataset(balanced_data)
    
    # Print dataset statistics
    print("\nDataset Shape:", balanced_data.shape)
    print("\nTraining set shape:", splits['X_train'].shape)
    print("Validation set shape:", splits['X_val'].shape)
    print("Test set shape:", splits['X_test'].shape)
    
    # Save the datasets
    print("\nSaving datasets...")
    balanced_data.to_csv('ckd_enhanced_dataset.csv', index=False)
    
    # Save train, validation, and test sets
    train_data = pd.concat([splits['X_train'], splits['y_train']], axis=1)
    val_data = pd.concat([splits['X_val'], splits['y_val']], axis=1)
    test_data = pd.concat([splits['X_test'], splits['y_test']], axis=1)
    
    train_data.to_csv('ckd_train_set.csv', index=False)
    val_data.to_csv('ckd_validation_set.csv', index=False)
    test_data.to_csv('ckd_test_set.csv', index=False)
    
    print("\nDataset generation and processing complete!")

if __name__ == "__main__":
    main()
