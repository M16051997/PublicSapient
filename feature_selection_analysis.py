# Feature Selection: Correlation & Multicollinearity Analysis
# Loan Default Prediction Project
# Date: 2025-11-03

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("✅ Libraries imported successfully")

# ============================================================================
# STEP 1: LOAD PROCESSED DATA
# ============================================================================
print("="*80)
print("STEP 1: LOADING PROCESSED DATA")
print("="*80)

# Load the processed data
X = pd.read_csv('processed_data/X_processed.csv')
y = pd.read_csv('processed_data/y_target.csv').squeeze()

print(f"\n✓ Features loaded: {X.shape}")
print(f"✓ Target loaded: {y.shape}")
print(f"✓ Default rate: {y.mean():.2%}")

# ============================================================================
# STEP 2: INITIAL FEATURE OVERVIEW
# ============================================================================
print("\n" + "="*80)
print("STEP 2: FEATURE OVERVIEW")
print("="*80)

# Separate features by type
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"\n📊 Feature Types:")
print(f"  - Numeric features: {len(numeric_features)}")
print(f"  - Categorical features: {len(categorical_features)}")
print(f"  - Total features: {X.shape[1]}")

# Check for constant features
constant_features = []
for col in X.columns:
    if X[col].nunique() <= 1:
        constant_features.append(col)

if constant_features:
    print(f"\n⚠️  Found {len(constant_features)} constant features to remove:")
    for feat in constant_features:
        print(f"    - {feat}")
    X = X.drop(columns=constant_features)
    numeric_features = [f for f in numeric_features if f not in constant_features]
else:
    print("\n✓ No constant features found")

# ============================================================================
# STEP 3: CORRELATION ANALYSIS (Numeric Features)
# ============================================================================
print("\n" + "="*80)
print("STEP 3: CORRELATION ANALYSIS")
print("="*80)

# Calculate correlation matrix for numeric features
numeric_data = X[numeric_features]
corr_matrix = numeric_data.corr()

# Find highly correlated feature pairs
def find_high_correlations(corr_matrix, threshold=0.8):
    """Find feature pairs with correlation above threshold"""
    high_corr_pairs = []
    
    # Get upper triangle of correlation matrix
    upper_triangle = np.triu(corr_matrix, k=1)
    
    # Find indices where correlation > threshold
    high_corr_indices = np.where(np.abs(upper_triangle) > threshold)
    
    # Get feature names for high correlations
    for i, j in zip(high_corr_indices[0], high_corr_indices[1]):
        feature1 = corr_matrix.columns[i]
        feature2 = corr_matrix.columns[j]
        correlation = corr_matrix.iloc[i, j]
        high_corr_pairs.append({
            'Feature_1': feature1,
            'Feature_2': feature2,
            'Correlation': correlation
        })
    
    return pd.DataFrame(high_corr_pairs).sort_values('Correlation', 
                                                     ascending=False, 
                                                     key=abs)

# Find high correlations (increased threshold to 0.9)
high_corr_df = find_high_correlations(corr_matrix, threshold=0.9)

if len(high_corr_df) > 0:
    print(f"\n⚠️  Found {len(high_corr_df)} highly correlated feature pairs (|r| > 0.9):")
    print("\n" + high_corr_df.to_string(index=False))
else:
    print("\n✓ No highly correlated feature pairs found (|r| > 0.9)")

# Visualize correlation matrix
plt.figure(figsize=(20, 16))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1, annot=False)
plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('feature_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Correlation matrix saved to: feature_correlation_matrix.png")

# ============================================================================
# STEP 4: TARGET CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("STEP 4: TARGET CORRELATION ANALYSIS")
print("="*80)

# Calculate correlation with target
target_corr = numeric_data.corrwith(y).abs().sort_values(ascending=False)

print("\n📊 Top 20 Features Correlated with Default:")
for i, (feature, corr) in enumerate(target_corr.head(20).items(), 1):
    print(f"{i:2d}. {feature:35s} |r| = {corr:.4f}")

# Identify low correlation features (reduced threshold to 0.001)
low_corr_threshold = 0.001
low_corr_features = target_corr[target_corr < low_corr_threshold].index.tolist()

if low_corr_features:
    print(f"\n⚠️  Found {len(low_corr_features)} features with very low target correlation (|r| < {low_corr_threshold}):")
    for feat in low_corr_features[:10]:  # Show first 10
        print(f"    - {feat}: |r| = {target_corr[feat]:.4f}")
    if len(low_corr_features) > 10:
        print(f"    ... and {len(low_corr_features) - 10} more")

# Visualize target correlations
plt.figure(figsize=(10, 12))
target_corr.head(30).plot(kind='barh', color='steelblue')
plt.xlabel('Absolute Correlation with Default')
plt.title('Top 30 Features by Target Correlation', fontsize=14)
plt.tight_layout()
plt.savefig('target_correlation_top30.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Target correlation plot saved to: target_correlation_top30.png")

# ============================================================================
# STEP 5: MULTICOLLINEARITY ANALYSIS (VIF)
# ============================================================================
print("\n" + "="*80)
print("STEP 5: MULTICOLLINEARITY ANALYSIS (VIF)")
print("="*80)
print("Note: VIF > 10 indicates high multicollinearity")
print("      VIF > 5 indicates moderate multicollinearity")

# Calculate VIF for numeric features
print("\n⏳ Calculating VIF (this may take a minute)...")

def calculate_vif(df):
    """Calculate VIF for all numeric features"""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) 
                       for i in range(len(df.columns))]
    return vif_data.sort_values('VIF', ascending=False)

# Remove any remaining NaN or infinite values for VIF calculation
numeric_data_clean = numeric_data.fillna(0)
numeric_data_clean = numeric_data_clean.replace([np.inf, -np.inf], 0)

try:
    vif_df = calculate_vif(numeric_data_clean)
    
    # Show high VIF features
    high_vif = vif_df[vif_df['VIF'] > 10]
    moderate_vif = vif_df[(vif_df['VIF'] > 5) & (vif_df['VIF'] <= 10)]
    
    if len(high_vif) > 0:
        print(f"\n⚠️  Features with HIGH multicollinearity (VIF > 10):")
        for _, row in high_vif.iterrows():
            print(f"    - {row['Feature']:35s} VIF = {row['VIF']:,.2f}")
    
    if len(moderate_vif) > 0:
        print(f"\n⚠️  Features with MODERATE multicollinearity (VIF > 5):")
        for _, row in moderate_vif.head(10).iterrows():
            print(f"    - {row['Feature']:35s} VIF = {row['VIF']:,.2f}")
        if len(moderate_vif) > 10:
            print(f"    ... and {len(moderate_vif) - 10} more")
    
    if len(high_vif) == 0 and len(moderate_vif) == 0:
        print("\n✓ No significant multicollinearity detected (all VIF < 5)")
    
    # Save VIF results
    vif_df.to_csv('vif_analysis.csv', index=False)
    print("\n✓ VIF analysis saved to: vif_analysis.csv")
    
except Exception as e:
    print(f"\n⚠️  VIF calculation failed: {str(e)}")
    print("   This might be due to perfect multicollinearity.")
    vif_df = pd.DataFrame()

# ============================================================================
# STEP 6: MUTUAL INFORMATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("STEP 6: MUTUAL INFORMATION ANALYSIS")
print("="*80)

# Calculate mutual information for all numeric features
print("\n⏳ Calculating mutual information scores...")

mi_scores = mutual_info_classif(numeric_data_clean, y, random_state=42)
mi_df = pd.DataFrame({
    'Feature': numeric_features,
    'MI_Score': mi_scores
}).sort_values('MI_Score', ascending=False)

print("\n📊 Top 20 Features by Mutual Information:")
for i, row in mi_df.head(20).iterrows():
    print(f"{row.name + 1:2d}. {row['Feature']:35s} MI = {row['MI_Score']:.4f}")

# Features with zero mutual information
zero_mi = mi_df[mi_df['MI_Score'] == 0]['Feature'].tolist()
if zero_mi:
    print(f"\n⚠️  Found {len(zero_mi)} features with zero mutual information:")
    for feat in zero_mi[:10]:
        print(f"    - {feat}")

# ============================================================================
# STEP 7: FEATURE SELECTION RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("STEP 7: FEATURE SELECTION RECOMMENDATIONS")
print("="*80)

# Define business-critical features that should NEVER be removed
business_critical_features = {
    'Credit_Bureau', 
    'Active_Loan', 
    'House_Own', 
    'Client_Family_Members',
    'Has_Credit_Bureau_Record',
    'Employment_Years',
    'Client_Income_Type',
    'Is_Unemployed',
    'Credit_Income_Ratio',
    'Annuity_Income_Ratio',
    'Monthly_Payment_Burden',
    'Total_Assets_Owned',
    'Age_Years',
    'Default'  # Target variable
}

print("\n🛡️ Business-critical features that will be protected:")
protected_features = [f for f in business_critical_features if f in X.columns]
for i, feat in enumerate(protected_features[:10], 1):
    print(f"   {i}. {feat}")
if len(protected_features) > 10:
    print(f"   ... and {len(protected_features) - 10} more")

# Collect features to potentially remove
features_to_remove = set()
removal_reasons = {}

# 1. Highly correlated pairs (keep the one with higher target correlation)
if len(high_corr_df) > 0:
    print("\n1️⃣ Handling Highly Correlated Pairs:")
    for _, row in high_corr_df.iterrows():
        feat1, feat2 = row['Feature_1'], row['Feature_2']
        corr1 = abs(target_corr.get(feat1, 0))
        corr2 = abs(target_corr.get(feat2, 0))
        
        # Check if either feature is business critical
        if feat1 in business_critical_features and feat2 in business_critical_features:
            print(f"   - Both '{feat1}' and '{feat2}' are business critical - keeping both")
            continue
        elif feat1 in business_critical_features:
            features_to_remove.add(feat2)
            removal_reasons[feat2] = f"High correlation with {feat1} (r={row['Correlation']:.3f})"
            print(f"   - Remove '{feat2}' (keep '{feat1}' - business critical)")
        elif feat2 in business_critical_features:
            features_to_remove.add(feat1)
            removal_reasons[feat1] = f"High correlation with {feat2} (r={row['Correlation']:.3f})"
            print(f"   - Remove '{feat1}' (keep '{feat2}' - business critical)")
        else:
            # Original logic if neither is business critical
            if corr1 > corr2:
                features_to_remove.add(feat2)
                removal_reasons[feat2] = f"High correlation with {feat1} (r={row['Correlation']:.3f})"
                print(f"   - Remove '{feat2}' (keep '{feat1}')")
            else:
                features_to_remove.add(feat1)
                removal_reasons[feat1] = f"High correlation with {feat2} (r={row['Correlation']:.3f})"
                print(f"   - Remove '{feat1}' (keep '{feat2}')")

# 2. High VIF features (if VIF > 10)
if 'vif_df' in locals() and len(vif_df) > 0:
    high_vif_features = vif_df[vif_df['VIF'] > 10]['Feature'].tolist()
    if high_vif_features:
        print("\n2️⃣ High VIF Features to Consider:")
        for feat in high_vif_features[:5]:
            if feat not in features_to_remove and feat not in business_critical_features:
                print(f"   - Consider removing '{feat}' (VIF = {vif_df[vif_df['Feature']==feat]['VIF'].values[0]:.1f})")

# 3. Low importance features (reduced MI threshold to 0.001)
low_importance_features = set()
if len(low_corr_features) > 0:
    # Features with both low correlation and low MI
    low_mi_features = mi_df[mi_df['MI_Score'] < 0.001]['Feature'].tolist()
    low_importance_features = set(low_corr_features) & set(low_mi_features)
    
    # Remove business critical features from the low importance set
    low_importance_features = low_importance_features - business_critical_features
    
    if low_importance_features:
        print(f"\n3️⃣ Low Importance Features (low correlation AND low MI):")
        for feat in list(low_importance_features)[:10]:
            features_to_remove.add(feat)
            removal_reasons[feat] = "Low correlation and mutual information"
            print(f"   - Remove '{feat}'")
        if len(low_importance_features) > 10:
            print(f"   ... and {len(low_importance_features) - 10} more")

# Summary
print(f"\n📋 FEATURE SELECTION SUMMARY:")
print(f"   - Total features: {X.shape[1]}")
print(f"   - Business-critical features protected: {len(protected_features)}")
print(f"   - Features to remove: {len(features_to_remove)}")
print(f"   - Features to keep: {X.shape[1] - len(features_to_remove)}")

# Create final feature list
features_to_keep = [col for col in X.columns if col not in features_to_remove]

# Save recommendations
recommendations = pd.DataFrame([
    {'Feature': feat, 'Action': 'Remove', 'Reason': removal_reasons.get(feat, 'Multiple reasons')}
    for feat in features_to_remove
])

if len(recommendations) > 0:
    recommendations.to_csv('feature_removal_recommendations.csv', index=False)
    print(f"\n✓ Recommendations saved to: feature_removal_recommendations.csv")

# ============================================================================
# STEP 8: CREATE CLEANED FEATURE SET
# ============================================================================
print("\n" + "="*80)
print("STEP 8: CREATING CLEANED FEATURE SET")
print("="*80)

# Remove recommended features
X_selected = X[features_to_keep].copy()

print(f"\n✓ Original features: {X.shape[1]}")
print(f"✓ Selected features: {X_selected.shape[1]}")
print(f"✓ Features removed: {len(features_to_remove)}")

# Save selected features
X_selected.to_csv('processed_data/X_selected.csv', index=False)
print(f"\n✓ Selected features saved to: processed_data/X_selected.csv")

# Save feature list
with open('processed_data/selected_features.txt', 'w') as f:
    f.write("SELECTED FEATURES AFTER CORRELATION/MULTICOLLINEARITY ANALYSIS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Total features kept: {len(features_to_keep)}\n\n")
    
    # Separate by type
    selected_numeric = [f for f in features_to_keep if f in numeric_features]
    selected_categorical = [f for f in features_to_keep if f in categorical_features]
    
    f.write(f"Numeric features ({len(selected_numeric)}):\n")
    for i, feat in enumerate(selected_numeric, 1):
        f.write(f"{i:3d}. {feat}\n")
    
    f.write(f"\nCategorical features ({len(selected_categorical)}):\n")
    for i, feat in enumerate(selected_categorical, 1):
        f.write(f"{i:3d}. {feat}\n")

print("✓ Feature list saved to: processed_data/selected_features.txt")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✅ FEATURE SELECTION COMPLETE!")
print("="*80)
print(f"""
📊 Analysis Results:
   - Highly correlated pairs found: {len(high_corr_df)}
   - Features with high VIF: {len(vif_df[vif_df['VIF'] > 10]) if 'vif_df' in locals() else 'N/A'}
   - Low importance features: {len(low_importance_features)}
   
📈 Feature Reduction:
   - Original features: {X.shape[1]}
   - Features removed: {len(features_to_remove)}
   - Final features: {X_selected.shape[1]}
   - Reduction: {(len(features_to_remove)/X.shape[1])*100:.1f}%

📁 Output Files:
   - feature_correlation_matrix.png
   - target_correlation_top30.png
   - vif_analysis.csv
   - feature_removal_recommendations.csv
   - processed_data/X_selected.csv
   - processed_data/selected_features.txt

🎯 Next Steps:
   1. Review the removed features to ensure business logic
   2. Proceed with train/test split
   3. Apply encoding to categorical variables
   4. Train models with selected features
""")

print("\n" + "="*80)