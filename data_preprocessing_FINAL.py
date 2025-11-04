"""
FINAL Data Preprocessing & Feature Engineering Pipeline
Based on Actual Data Type Analysis
Loan Default Prediction - Production Ready

Column Types Identified from Diagnostics:
- 9 object columns that are 100% numeric (need conversion)
- 5 binary flags (float) that should be int
- 2 Yes/No columns that need binary conversion
- 3 categorical columns with anomalies ('##', 'XNA')
- 11 truly categorical columns
- 15 already-correct numeric columns

Problematic values found in data:
- '$' in Client_Income, Credit_Amount, Loan_Annuity
- '#VALUE!' in Loan_Annuity
- '@', '#' in Population_Region_Relative
- 'x' in Age_Days, Employed_Days, Registration_Days, ID_Days
- '&' in Score_Source_3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# COLUMN TYPE MAPPING (from diagnostic results)
# ============================================================================

COLUMN_TYPES = {
    # Objects that should be numeric (100% convertible after cleaning)
    'force_numeric': [
        'Client_Income',
        'Credit_Amount',
        'Loan_Annuity',
        'Population_Region_Relative',
        'Age_Days',
        'Employed_Days',
        'Registration_Days',
        'ID_Days',
        'Score_Source_3'
    ],
    
    # Continuous numeric (already correct)
    'continuous_numeric': [
        'ID',
        'Child_Count',
        'Own_House_Age',
        'Client_Family_Members',
        'Cleint_City_Rating',
        'Application_Process_Day',
        'Application_Process_Hour',
        'Score_Source_1',
        'Score_Source_2',
        'Social_Circle_Default',
        'Phone_Change',
        'Credit_Bureau'
    ],
    
    # Binary flags (0/1)
    'binary_flags': [
        'Car_Owned',
        'Bike_Owned',
        'Active_Loan',
        'House_Own',
        'Mobile_Tag',
        'Homephone_Tag',
        'Workphone_Working'
    ],
    
    # Categorical
    'categorical': [
        'Accompany_Client',
        'Client_Income_Type',
        'Client_Education',
        'Client_Marital_Status',
        'Client_Gender',
        'Loan_Contract_Type',
        'Client_Housing_Type',
        'Client_Occupation',
        'Client_Permanent_Match_Tag',
        'Client_Contact_Work_Tag',
        'Type_Organization'
    ]
}

# Known problematic values from diagnostics
PROBLEMATIC_VALUES = {
    'Client_Income': ['$'],
    'Credit_Amount': ['$'],
    'Loan_Annuity': ['$', '#VALUE!'],
    'Population_Region_Relative': ['@', '#'],
    'Age_Days': ['x'],
    'Employed_Days': ['x'],
    'Registration_Days': ['x'],
    'ID_Days': ['x'],
    'Score_Source_3': ['&']
}


# ============================================================================
# SECTION 1: DATA LOADING & INSPECTION
# ============================================================================

def load_and_inspect_data(filepath):
    """Load data and perform initial inspection"""
    print("="*80)
    print("STEP 1: DATA LOADING & INITIAL INSPECTION")
    print("="*80)
    
    dt = pd.read_csv(filepath, low_memory=False)
    
    print(f"\n✓ Dataset loaded: {dt.shape[0]:,} rows × {dt.shape[1]} columns")
    print(f"✓ Memory usage: {dt.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Target distribution
    if 'Default' in dt.columns:
        target_counts = dt['Default'].value_counts()
        print(f"\n✓ Target Distribution:")
        print(f"  - Non-Default (0): {target_counts[0]:,} ({target_counts[0]/len(dt)*100:.1f}%)")
        print(f"  - Default (1): {target_counts[1]:,} ({target_counts[1]/len(dt)*100:.1f}%)")
        print(f"  - Imbalance Ratio: {target_counts[0]/target_counts[1]:.1f}:1")
    
    return dt


# ============================================================================
# SECTION 2: CLEAN PROBLEMATIC VALUES (NEW!)
# ============================================================================

def clean_problematic_values(dt):
    """Clean known problematic characters before type conversion"""
    print("\n" + "="*80)
    print("STEP 2: CLEANING PROBLEMATIC VALUES")
    print("="*80)
    
    dt_clean = dt.copy()
    total_cleaned = 0
    
    print("\n✓ Removing problematic characters from numeric columns...")
    
    for col, bad_values in PROBLEMATIC_VALUES.items():
        if col not in dt_clean.columns:
            continue
        
        cleaned_count = 0
        for bad_val in bad_values:
            # Count occurrences
            if dt_clean[col].dtype == 'object':
                mask = dt_clean[col] == bad_val
                count = mask.sum()
                if count > 0:
                    # Replace with NaN
                    dt_clean.loc[mask, col] = np.nan
                    cleaned_count += count
        
        if cleaned_count > 0:
            print(f"  - {col}: {cleaned_count} problematic values ('{','.join(bad_values)}') → NaN")
            total_cleaned += cleaned_count
    
    print(f"\n✓ Total problematic values cleaned: {total_cleaned}")
    
    return dt_clean


# ============================================================================
# SECTION 3: DATA TYPE STANDARDIZATION
# ============================================================================

def standardize_data_types(dt):
    """Convert columns to correct types"""
    print("\n" + "="*80)
    print("STEP 3: DATA TYPE STANDARDIZATION")
    print("="*80)
    
    dt_clean = dt.copy()
    
    # 3.1 Convert object columns to numeric (should be 100% clean now)
    print(f"\n✓ Converting {len(COLUMN_TYPES['force_numeric'])} object columns to numeric...")
    for col in COLUMN_TYPES['force_numeric']:
        if col in dt_clean.columns:
            before_nulls = dt_clean[col].isnull().sum()
            dt_clean[col] = pd.to_numeric(dt_clean[col], errors='coerce')
            after_nulls = dt_clean[col].isnull().sum()
            new_nulls = after_nulls - before_nulls
            
            if new_nulls > 0:
                print(f"  ⚠️  {col}: {new_nulls} additional non-numeric values coerced to NaN")
            else:
                print(f"  ✓ {col}: Converted successfully (no additional NaNs)")
    
    # 3.2 Ensure binary flags are int (not float)
    print(f"\n✓ Standardizing {len(COLUMN_TYPES['binary_flags'])} binary flag columns...")
    for col in COLUMN_TYPES['binary_flags']:
        if col in dt_clean.columns:
            # Fill NaN with 0 and convert to int
            dt_clean[col] = dt_clean[col].fillna(0).astype(int)
            print(f"  - {col}: Standardized to int (NaN→0)")
    
    # 3.3 Handle Yes/No categorical columns
    yes_no_cols = ['Client_Permanent_Match_Tag', 'Client_Contact_Work_Tag']
    
    print(f"\n✓ Converting {len(yes_no_cols)} Yes/No columns to binary...")
    for col in yes_no_cols:
        if col in dt_clean.columns:
            dt_clean[col] = dt_clean[col].map({'Yes': 1, 'No': 0})
            dt_clean[col] = dt_clean[col].fillna(0).astype(int)
            print(f"  - {col}: Yes→1, No→0")
    
    # 3.4 Handle categorical anomalies
    print(f"\n✓ Cleaning categorical anomalies...")
    
    anomaly_replacements = {
        'Accompany_Client': ['##'],
        'Client_Gender': ['XNA'],
        'Type_Organization': ['XNA']
    }
    
    for col, anomalies in anomaly_replacements.items():
        if col in dt_clean.columns:
            before = dt_clean[col].isnull().sum()
            for anomaly in anomalies:
                dt_clean[col] = dt_clean[col].replace(anomaly, np.nan)
            after = dt_clean[col].isnull().sum()
            if after > before:
                print(f"  - {col}: {after - before} anomalies ('{','.join(anomalies)}') → NaN")
    
    # 3.5 Handle sentinel values
    print(f"\n✓ Handling sentinel values...")
    if 'Employed_Days' in dt_clean.columns:
        sentinel_count = (dt_clean['Employed_Days'] == 365243).sum()
        if sentinel_count > 0:
            dt_clean['Employed_Days'] = dt_clean['Employed_Days'].replace(365243, np.nan)
            print(f"  - Employed_Days: {sentinel_count:,} sentinel values (365243) → NaN")
    
    # 3.6 Ensure days columns are negative
    days_columns = ['Age_Days', 'Employed_Days', 'Registration_Days', 'ID_Days']
    
    print(f"\n✓ Ensuring days columns are negative...")
    for col in days_columns:
        if col in dt_clean.columns:
            # Make positive values negative
            positive_count = (dt_clean[col] > 0).sum()
            if positive_count > 0:
                dt_clean[col] = -dt_clean[col].abs()
                print(f"  - {col}: {positive_count:,} positive values → negative")
    
    print(f"\n✓ Data type standardization complete!")
    print(f"  Final dtypes: {dt_clean.dtypes.value_counts().to_dict()}")
    
    return dt_clean


# ============================================================================
# SECTION 4: OUTLIER DETECTION & TREATMENT
# ============================================================================

def detect_and_treat_outliers(dt):
    """Detect and handle outliers in continuous numeric columns"""
    print("\n" + "="*80)
    print("STEP 4: OUTLIER DETECTION & TREATMENT")
    print("="*80)
    
    dt_clean = dt.copy()
    
    # Get continuous numeric columns (exclude binary flags, ID, target)
    outlier_check_cols = []
    for col in COLUMN_TYPES['continuous_numeric'] + COLUMN_TYPES['force_numeric']:
        if col in dt_clean.columns and col != 'ID':
            # Only check if >10 unique values
            if dt_clean[col].nunique() > 10:
                outlier_check_cols.append(col)
    
    print(f"\n✓ Checking {len(outlier_check_cols)} continuous columns for outliers...")
    
    outlier_summary = []
    
    for col in outlier_check_cols:
        if dt_clean[col].notna().sum() < 10:
            continue
        
        # IQR method
        Q1 = dt_clean[col].quantile(0.25)
        Q3 = dt_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outliers = ((dt_clean[col] < lower_bound) | (dt_clean[col] > upper_bound))
        outlier_count = outliers.sum()
        outlier_pct = (outlier_count / len(dt_clean)) * 100
        
        if outlier_count > 0 and outlier_pct > 0.1:
            outlier_summary.append({
                'Column': col,
                'Count': outlier_count,
                'Percentage': f"{outlier_pct:.2f}%",
                'Capped_Lower': f"{lower_bound:.2f}",
                'Capped_Upper': f"{upper_bound:.2f}"
            })
            
            # Cap outliers
            dt_clean[col] = dt_clean[col].clip(lower=lower_bound, upper=upper_bound)
    
    if outlier_summary:
        outlier_df = pd.DataFrame(outlier_summary)
        print("\n✓ Outlier Detection & Capping Summary:")
        print(outlier_df.to_string(index=False))
    else:
        print("\n✓ No significant outliers detected")
    
    return dt_clean


# ============================================================================
# SECTION 5: MISSING VALUE ANALYSIS & IMPUTATION
# ============================================================================

def handle_missing_values(dt):
    """Comprehensive missing value handling"""
    print("\n" + "="*80)
    print("STEP 5: MISSING VALUE ANALYSIS & IMPUTATION")
    print("="*80)
    
    dt_clean = dt.copy()
    
    # 5.1 Missing value summary
    missing_summary = []
    for col in dt_clean.columns:
        missing_count = dt_clean[col].isnull().sum()
        if missing_count > 0:
            missing_summary.append({
                'Column': col,
                'Missing': missing_count,
                'Percentage': f"{(missing_count/len(dt_clean)*100):.2f}%",
                'Type': str(dt_clean[col].dtype)
            })
    
    if missing_summary:
        missing_df = pd.DataFrame(missing_summary).sort_values('Missing', ascending=False)
        print("\n✓ Missing Value Summary:")
        print(missing_df.to_string(index=False))
    
    # 5.2 Create missing indicators for columns with >5% missing
    print(f"\n✓ Creating missing indicators for columns with >5% missing...")
    missing_flag_count = 0
    for col in dt_clean.columns:
        if col == 'Default':
            continue
        missing_pct = (dt_clean[col].isnull().sum() / len(dt_clean)) * 100
        if missing_pct > 5:
            dt_clean[f'{col}_missing_flag'] = dt_clean[col].isnull().astype(int)
            print(f"  - {col}_missing_flag created ({missing_pct:.1f}% missing)")
            missing_flag_count += 1
    
    if missing_flag_count > 0:
        print(f"✓ Total missing indicators created: {missing_flag_count}")
    
    # 5.3 Numeric imputation (median)
    numeric_cols = (
        COLUMN_TYPES['force_numeric'] + 
        COLUMN_TYPES['continuous_numeric'] +
        COLUMN_TYPES['binary_flags']
    )
    numeric_impute_cols = [col for col in numeric_cols 
                          if col in dt_clean.columns and col != 'ID' and dt_clean[col].isnull().sum() > 0]
    
    if numeric_impute_cols:
        print(f"\n✓ Imputing {len(numeric_impute_cols)} numeric columns with median...")
        for col in numeric_impute_cols:
            median_val = dt_clean[col].median()
            before_nulls = dt_clean[col].isnull().sum()
            dt_clean[col].fillna(median_val, inplace=True)
            print(f"  - {col}: {before_nulls:,} → median ({median_val:.4f})")
    
    # 5.4 Categorical imputation ('Unknown')
    categorical_impute_cols = [col for col in COLUMN_TYPES['categorical'] 
                               if col in dt_clean.columns and dt_clean[col].isnull().sum() > 0]
    
    if categorical_impute_cols:
        print(f"\n✓ Imputing {len(categorical_impute_cols)} categorical columns with 'Unknown'...")
        for col in categorical_impute_cols:
            before_nulls = dt_clean[col].isnull().sum()
            if dt_clean[col].dtype.name != 'category':
                dt_clean[col] = dt_clean[col].astype('category')
            if 'Unknown' not in dt_clean[col].cat.categories:
                dt_clean[col] = dt_clean[col].cat.add_categories('Unknown')
            dt_clean[col].fillna('Unknown', inplace=True)
            print(f"  - {col}: {before_nulls:,} → 'Unknown'")
    
    # Final check
    remaining_nulls = dt_clean.isnull().sum().sum()
    print(f"\n✓ Total remaining missing values: {remaining_nulls}")
    
    return dt_clean


# ============================================================================
# SECTION 6: FEATURE ENGINEERING
# ============================================================================

def engineer_features(dt):
    """Create derived features based on EDA insights"""
    print("\n" + "="*80)
    print("STEP 6: FEATURE ENGINEERING")
    print("="*80)
    
    dt_feat = dt.copy()
    feature_count = 0
    
    # 6.1 Convert days to years
    print("\n✓ Converting days to years...")
    dt_feat['Age_Years'] = (-dt_feat['Age_Days'] / 365).round(1)
    dt_feat['Employment_Years'] = (-dt_feat['Employed_Days'] / 365).round(1)
    dt_feat['Registration_Years'] = (-dt_feat['Registration_Days'] / 365).round(1)
    dt_feat['ID_Change_Years'] = (-dt_feat['ID_Days'] / 365).round(1)
    dt_feat['Phone_Change_Years'] = (dt_feat['Phone_Change'] / 365).round(1)
    feature_count += 5
    
    # 6.2 Financial ratios
    print("\n✓ Creating financial ratio features...")
    dt_feat['Credit_Income_Ratio'] = dt_feat['Credit_Amount'] / (dt_feat['Client_Income'] + 1)
    dt_feat['Annuity_Income_Ratio'] = dt_feat['Loan_Annuity'] / (dt_feat['Client_Income'] + 1)
    dt_feat['Credit_Annuity_Ratio'] = dt_feat['Credit_Amount'] / (dt_feat['Loan_Annuity'] + 1)
    dt_feat['Monthly_Payment_Burden'] = (dt_feat['Loan_Annuity'] * 12) / (dt_feat['Client_Income'] + 1)
    feature_count += 4
    
    # 6.3 Risk flags from EDA
    print("\n✓ Creating risk indicator flags...")
    dt_feat['Is_Unemployed'] = (dt_feat['Client_Income_Type'] == 'Unemployed').astype(int)
    dt_feat['High_Risk_Income_Type'] = dt_feat['Client_Income_Type'].isin(['Unemployed', 'Student']).astype(int)
    dt_feat['Is_Revolving_Loan'] = (dt_feat['Loan_Contract_Type'] == 'RL').astype(int)
    dt_feat['Applied_Alone'] = (dt_feat['Accompany_Client'] == 'Alone').astype(int)
    dt_feat['Unstable_Housing'] = dt_feat['Client_Housing_Type'].isin(['Shared', 'Rental']).astype(int)
    dt_feat['Low_Education'] = dt_feat['Client_Education'].isin(['Secondary', 'Junior secondary']).astype(int)
    feature_count += 6
    
    # 6.4 Asset ownership
    print("\n✓ Creating asset ownership features...")
    dt_feat['Total_Assets_Owned'] = dt_feat['Car_Owned'] + dt_feat['Bike_Owned'] + dt_feat['House_Own']
    dt_feat['Has_No_Assets'] = (dt_feat['Total_Assets_Owned'] == 0).astype(int)
    feature_count += 2
    
    # 6.5 Stability indicators
    print("\n✓ Creating stability features...")
    dt_feat['Job_Stability'] = (dt_feat['Employment_Years'] > 5).astype(int)
    dt_feat['Address_Stability'] = (dt_feat['Registration_Years'] > 3).astype(int)
    dt_feat['Age_Group'] = pd.cut(dt_feat['Age_Years'], 
                                   bins=[0, 25, 35, 45, 55, 100],
                                   labels=['18-25', '26-35', '36-45', '46-55', '55+'])
    feature_count += 3
    
    # 6.6 Credit behavior
    print("\n✓ Creating credit behavior features...")
    dt_feat['Has_Credit_History'] = (dt_feat['Credit_Bureau'] > 0).astype(int)
    dt_feat['High_Credit_Inquiries'] = (dt_feat['Credit_Bureau'] > 3).astype(int)
    dt_feat['Social_Default_Risk'] = (dt_feat['Social_Circle_Default'] > 0).astype(int)
    feature_count += 3
    
    # 6.7 External score aggregations
    print("\n✓ Creating score aggregation features...")
    score_cols = ['Score_Source_1', 'Score_Source_2', 'Score_Source_3']
    dt_feat['Avg_External_Score'] = dt_feat[score_cols].mean(axis=1)
    dt_feat['Min_External_Score'] = dt_feat[score_cols].min(axis=1)
    dt_feat['Max_External_Score'] = dt_feat[score_cols].max(axis=1)
    dt_feat['Score_Range'] = dt_feat['Max_External_Score'] - dt_feat['Min_External_Score']
    feature_count += 4
    
    # 6.8 Contact verification
    print("\n✓ Creating contact verification features...")
    dt_feat['Contact_Verification_Score'] = (
        dt_feat['Mobile_Tag'] + dt_feat['Homephone_Tag'] + dt_feat['Workphone_Working']
    )
    dt_feat['Full_Contact_Verified'] = (dt_feat['Contact_Verification_Score'] == 3).astype(int)
    feature_count += 2
    
    # 6.9 Family burden
    print("\n✓ Creating family burden features...")
    dt_feat['High_Child_Count'] = (dt_feat['Child_Count'] >= 3).astype(int)
    dt_feat['Dependents_Per_Income'] = dt_feat['Child_Count'] / (dt_feat['Client_Income'] / 10000 + 1)
    feature_count += 2
    
    # 6.10 Application timing
    print("\n✓ Creating temporal features...")
    dt_feat['Applied_Weekend'] = dt_feat['Application_Process_Day'].isin([0, 6]).astype(int)
    dt_feat['Applied_Business_Hours'] = dt_feat['Application_Process_Hour'].between(9, 17).astype(int)
    feature_count += 2
    
    # 6.11 Document freshness
    print("\n✓ Creating document freshness features...")
    dt_feat['Recent_ID_Change'] = (dt_feat['ID_Change_Years'] < 1).astype(int)
    dt_feat['Recent_Phone_Change'] = (dt_feat['Phone_Change_Years'] < 1).astype(int)
    dt_feat['Frequent_Changes'] = dt_feat['Recent_ID_Change'] + dt_feat['Recent_Phone_Change']
    feature_count += 3
    
    # 6.12 Composite risk score
    print("\n✓ Creating composite risk score...")
    risk_components = [
        dt_feat['Is_Unemployed'] * 3,
        dt_feat['Has_No_Assets'] * 2,
        dt_feat['Applied_Alone'],
        dt_feat['Is_Revolving_Loan'],
        dt_feat['Unstable_Housing'],
        dt_feat['High_Risk_Income_Type'],
        dt_feat['Low_Education'],
        dt_feat['Social_Default_Risk']
    ]
    dt_feat['Composite_Risk_Score'] = sum(risk_components)
    feature_count += 1
    
    print(f"\n✅ Feature Engineering Complete: {feature_count} new features created!")
    
    return dt_feat


# ============================================================================
# SECTION 7: CATEGORICAL ENCODING
# ============================================================================

def encode_categorical_features(dt):
    """Encode categorical variables"""
    print("\n" + "="*80)
    print("STEP 7: CATEGORICAL ENCODING")
    print("="*80)
    
    dt_encoded = dt.copy()
    
    # 7.1 One-hot encoding for low-cardinality
    low_card_cols = ['Client_Gender', 'Loan_Contract_Type', 'Client_Marital_Status']
    existing_low_card = [col for col in low_card_cols if col in dt_encoded.columns]
    
    if existing_low_card:
        print(f"\n✓ One-hot encoding {len(existing_low_card)} low-cardinality columns...")
        for col in existing_low_card:
            print(f"  - {col}: {dt_encoded[col].nunique()} categories")
        dt_encoded = pd.get_dummies(dt_encoded, columns=existing_low_card, drop_first=True, prefix_sep='_')
    
    # 7.2 Frequency encoding for high-cardinality
    high_card_cols = ['Client_Occupation', 'Type_Organization']
    
    print(f"\n✓ Frequency encoding high-cardinality columns...")
    for col in high_card_cols:
        if col in dt_encoded.columns:
            freq_map = dt_encoded[col].value_counts(normalize=True)
            dt_encoded[f'{col}_Frequency'] = dt_encoded[col].map(freq_map)
            print(f"  - {col}: {len(freq_map)} categories → frequency")
            dt_encoded.drop(columns=[col], inplace=True)
    
    # 7.3 Label encoding for medium cardinality
    label_encode_cols = ['Client_Income_Type', 'Client_Education', 'Client_Housing_Type', 'Accompany_Client']
    
    print(f"\n✓ Label encoding medium-cardinality columns...")
    le = LabelEncoder()
    for col in label_encode_cols:
        if col in dt_encoded.columns:
            dt_encoded[f'{col}_LabelEncoded'] = le.fit_transform(dt_encoded[col].astype(str))
            print(f"  - {col}: {dt_encoded[col].nunique()} categories")
    
    # 7.4 Ordinal encoding for Age_Group
    if 'Age_Group' in dt_encoded.columns:
        age_mapping = {'18-25': 1, '26-35': 2, '36-45': 3, '46-55': 4, '55+': 5}
        dt_encoded['Age_Group_Ordinal'] = dt_encoded['Age_Group'].map(age_mapping)
        dt_encoded.drop(columns=['Age_Group'], inplace=True)
        print(f"\n✓ Age_Group → ordinal encoding")
    
    print(f"\n✅ Categorical encoding complete!")
    
    return dt_encoded


# ============================================================================
# SECTION 8: FINAL FEATURE PREPARATION
# ============================================================================

def prepare_final_features(dt):
    """Prepare final feature set for modeling"""
    print("\n" + "="*80)
    print("STEP 8: FINAL FEATURE PREPARATION")
    print("="*80)
    
    dt_final = dt.copy()
    
    # Drop unnecessary columns
    drop_cols = [
        'ID',
        'Age_Days', 'Employed_Days', 'Registration_Days', 'ID_Days', 'Phone_Change',
        'Client_Income_Type', 'Client_Education', 'Client_Housing_Type', 'Accompany_Client'
    ]
    
    existing_drop_cols = [col for col in drop_cols if col in dt_final.columns]
    if existing_drop_cols:
        dt_final.drop(columns=existing_drop_cols, inplace=True)
        print(f"\n✓ Dropped {len(existing_drop_cols)} unnecessary columns")
    
    # Separate features and target
    if 'Default' in dt_final.columns:
        X = dt_final.drop(columns=['Default'])
        y = dt_final['Default']
        print(f"\n✓ Features (X): {X.shape}")
        print(f"✓ Target (y): {y.shape}")
        print(f"✓ Total features: {X.shape[1]}")
    else:
        X = dt_final
        y = None
        print(f"\n✓ Features (X): {X.shape}")
    
    # Final data quality check
    print(f"\n✓ Final Data Quality Check:")
    print(f"  - Missing values: {X.isnull().sum().sum()}")
    print(f"  - Infinite values: {np.isinf(X.select_dtypes(include=[np.number])).sum().sum()}")
    print(f"  - Duplicate rows: {X.duplicated().sum()}")
    
    # Check for non-numeric
    non_numeric = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if non_numeric:
        print(f"\n⚠️  Non-numeric columns: {non_numeric}")
    else:
        print(f"\n✅ All features are numeric - ready for modeling!")
    
    return X, y


# ============================================================================
# SECTION 9: SAVE PROCESSED DATA
# ============================================================================

def save_processed_data(X, y, output_dir='processed_data'):
    """Save processed features and target"""
    import os
    
    print("\n" + "="*80)
    print("STEP 9: SAVING PROCESSED DATA")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save features
    X_path = os.path.join(output_dir, 'X_processed.csv')
    X.to_csv(X_path, index=False)
    print(f"\n✓ Features saved: {X_path}")
    
    # Save target
    if y is not None:
        y_path = os.path.join(output_dir, 'y_target.csv')
        y.to_csv(y_path, index=False, header=True)
        print(f"✓ Target saved: {y_path}")
    
    # Save feature list
    feature_list_path = os.path.join(output_dir, 'feature_list.txt')
    with open(feature_list_path, 'w') as f:
        for i, col in enumerate(X.columns, 1):
            f.write(f"{i}. {col}\n")
    print(f"✓ Feature list saved: {feature_list_path}")
    
    print(f"\n✅ All files saved to: {output_dir}/")


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def main(filepath):
    """Execute complete preprocessing pipeline"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*15 + "LOAN DEFAULT PREDICTION - FINAL V2" + " "*28 + "║")
    print("║" + " "*10 + "Enhanced with Explicit Problematic Value Cleaning" + " "*17 + "║")
    print("╚" + "="*78 + "╝")
    
    # Execute pipeline
    dt = load_and_inspect_data(filepath)
    dt = clean_problematic_values(dt)  # NEW STEP!
    dt = standardize_data_types(dt)
    dt = detect_and_treat_outliers(dt)
    dt = handle_missing_values(dt)
    dt = engineer_features(dt)
    dt = encode_categorical_features(dt)
    X, y = prepare_final_features(dt)
    save_processed_data(X, y)
    
    print("\n" + "="*80)
    print("✅ PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nReady for modeling!")
    print("="*80 + "\n")
    
    return X, y


if __name__ == "__main__":
    DATASET_PATH = 'data/Dataset.csv'
    
    X_processed, y_target = main(DATASET_PATH)
    
    print("\n📊 Final Summary:")
    print(f"  Features: {X_processed.shape[1]}")
    print(f"  Samples: {X_processed.shape[0]:,}")
    print(f"  All numeric: {len(X_processed.select_dtypes(include='object')) == 0}")
    print(f"  No missing: {X_processed.isnull().sum().sum() == 0}")
