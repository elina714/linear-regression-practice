import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (log_loss, roc_auc_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_curve, PrecisionRecallDisplay)
import joblib
import warnings

warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


# ------------------------------
# 1. Load and Explore Data
# ------------------------------
def load_and_explore(filepath):
    """Load data, perform basic checks and EDA."""
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Successfully loaded {filepath}")
    except FileNotFoundError:
        print(f"❌ Error: File {filepath} not found.")
        raise

    # Ensure target column exists and convert to integer
    if 'churn' not in df.columns:
        raise ValueError("Dataset must contain a 'churn' column.")
    df['churn'] = df['churn'].astype(int)

    print("\n" + "=" * 50)
    print("📊 DATA OVERVIEW")
    print("=" * 50)
    print(f"Data shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nTarget distribution:\n{df['churn'].value_counts(normalize=True)}")
    print(f"Churn Rate: {df['churn'].mean():.2%}")

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
        plt.title("Feature Correlations Heatmap")
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ Not enough numeric columns for correlation heatmap.")

    return df


# ------------------------------
# 2. Feature Selection
# ------------------------------
def prepare_features(df, feature_list=None):
    """
    Select features.
    NOTE: Missing values are handled in the pipeline to prevent data leakage.
    """
    if feature_list is None:
        feature_list = [col for col in df.columns
                        if col != 'churn' and pd.api.types.is_numeric_dtype(df[col])]

    # Filter to ensure all features exist in dataframe
    feature_list = [f for f in feature_list if f in df.columns]

    if not feature_list:
        raise ValueError("No valid features found!")

    print(f"\n✅ Using {len(feature_list)} features: {feature_list}")

    X = df[feature_list].copy()
    y = df['churn'].copy()

    # DO NOT fill missing values here - pipeline will handle it
    return X, y, feature_list


# ------------------------------
# 3. Build Pipeline and Tune Model
# ------------------------------
def train_model(X_train, y_train, feature_names, use_interactions=True):
    """
    Create pipeline with imputer, scaler, optional interactions, and logistic regression.
    Performs GridSearchCV to find optimal hyperparameters.
    """
    print("\n" + "=" * 50)
    print("🔧 MODEL TRAINING & HYPERPARAMETER TUNING")
    print("=" * 50)

    if use_interactions:
        print("📐 Including polynomial interactions (degree=2)...")
        pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ('clf', LogisticRegression(max_iter=1000, random_state=4))
        ])

        param_grid = {
            'poly__degree': [1, 2],  # 1 = no interactions, 2 = with interactions
            'clf__C': [0.005, 0.01, 0.1, 1, 10],
            'clf__penalty': ['l2'],  # l2 is more stable with interactions
            'clf__solver': ['liblinear'],
            'clf__class_weight': [None, 'balanced']
        }
    else:
        print("📐 Using original features only...")
        pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=4))
        ])

        param_grid = {
            'clf__C': [0.01, 0.1, 1, 10, 100],
            'clf__penalty': ['l1', 'l2'],
            'clf__solver': ['liblinear'],
            'clf__class_weight': [None, 'balanced']
        }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='neg_log_loss',  # Optimize for Log Loss
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\n✅ Best CV Log Loss: {-grid_search.best_score_:.4f}")
    print(f"✅ Best Parameters: {grid_search.best_params_}")

    return grid_search


# ------------------------------
# 4. Comprehensive Evaluation
# ------------------------------
def evaluate_model(model, X_train, y_train, X_test, y_test, feature_names=None):
    """Print various evaluation metrics and generate plots."""
    print("\n" + "=" * 50)
    print("📈 MODEL EVALUATION")
    print("=" * 50)

    # Predictions
    y_pred = model.predict(X_test)
    # After getting probabilities
    y_proba = model.predict_proba(X_test)[:, 1]

    # Try different thresholds
    for threshold in [0.5, 0.4, 0.3, 0.2]:
        y_pred_custom = (y_proba >= threshold).astype(int)
        print(f"\nThreshold {threshold}:")
        print(classification_report(y_test, y_pred_custom, target_names=['No Churn', 'Churn']))


    # --- Metrics ---
    test_log_loss = log_loss(y_test, y_proba)
    test_auc = roc_auc_score(y_test, y_proba)

    # Check for overfitting
    train_proba = model.predict_proba(X_train)[:, 1]
    train_log_loss = log_loss(y_train, train_proba)
    gap = test_log_loss - train_log_loss

    # Baseline comparison
    baseline_prob = np.full(len(y_test), y_train.mean())
    baseline_log_loss = log_loss(y_test, baseline_prob)
    improvement = (baseline_log_loss - test_log_loss) / baseline_log_loss

    print(f"\n🎯 Test Log Loss:      {test_log_loss:.4f}")
    print(f"🎯 Train Log Loss:     {train_log_loss:.4f}")
    print(f"📈 Overfitting Gap:    {gap:.4f} {'⚠️ High' if gap > 0.05 else '✅ OK'}")
    print(f"\n📊 Baseline Log Loss:  {baseline_log_loss:.4f}")
    print(f"🚀 Improvement:        {improvement:.1%} better than baseline")
    print(f"🎯 Test ROC AUC:       {test_auc:.4f}")

    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn'])
    disp.plot(ax=axes[0, 0], cmap='Blues')
    axes[0, 0].set_title("Confusion Matrix")

    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    axes[0, 1].plot(recall, precision, 'b-', linewidth=2)
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Probability Distribution
    axes[1, 0].hist(y_proba[y_test == 0], bins=20, alpha=0.5, label='No Churn', color='blue')
    axes[1, 0].hist(y_proba[y_test == 1], bins=20, alpha=0.5, label='Churn', color='red')
    axes[1, 0].axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    axes[1, 0].set_xlabel('Predicted Probability')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Probability Distribution')
    axes[1, 0].legend()

    # 4. ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[1, 1].plot(fpr, tpr, 'g-', linewidth=2, label=f'ROC Curve (AUC={test_auc:.2f})')
    axes[1, 1].plot([0, 1], [0, 1], 'r--', label='Random')
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('ROC Curve')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # --- Feature Coefficients ---
    try:
        clf = model.best_estimator_.named_steps['clf']
        coef = clf.coef_[0]

        # Get feature names after preprocessing
        if 'poly' in model.best_estimator_.named_steps:
            poly = model.best_estimator_.named_steps['poly']
            # Use original feature names to generate poly names
            final_feature_names = poly.get_feature_names_out(feature_names)
        else:
            final_feature_names = feature_names

        if len(coef) == len(final_feature_names):
            coef_series = pd.Series(coef, index=final_feature_names).sort_values()

            # Plot top 15 features to avoid clutter
            if len(coef_series) > 15:
                coef_series = pd.concat([coef_series.head(8), coef_series.tail(7)])

            plt.figure(figsize=(10, 6))
            coef_series.plot(kind='barh', color='steelblue')
            plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            plt.title("Top Feature Coefficients")
            plt.xlabel("Coefficient Value")
            plt.tight_layout()
            plt.show()
        else:
            print(f"⚠️ Coefficient length mismatch: {len(coef)} vs {len(final_feature_names)}")
    except Exception as e:
        print(f"⚠️ Could not plot coefficients: {e}")

    return test_log_loss, test_auc


# ------------------------------
# 5. Main Execution
# ------------------------------
def main():
    print("=" * 60)
    print("🚀 CHURN PREDICTION MODEL - LOGISTIC REGRESSION")
    print("=" * 60)

    # 1. Load Data
    df = load_and_explore('ChurnData.csv')

    # 2. Select Features
    # Update this list based on your actual CSV columns
    features = ['tenure', 'age', 'address', 'ed', 'equip', 'income', 'employ']
    X, y, feature_names = prepare_features(df, features)

    # 3. Split Data (Stratified to preserve churn ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=4, stratify=y
    )
    print(f"\n✅ Train set: {X_train.shape[0]} samples")
    print(f"✅ Test set:  {X_test.shape[0]} samples")

    # 4. Train Model
    # Set use_interactions=True to try polynomial features
    grid_search = train_model(X_train, y_train, feature_names, use_interactions=False)

    # 5. Evaluate
    test_log_loss, test_auc = evaluate_model(
        grid_search, X_train, y_train, X_test, y_test, feature_names=feature_names
    )

    # 6. Cross-Validation Stability Check
    cv_scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"\n🔄 CV ROC-AUC Scores: {cv_scores}")
    print(f"🔄 Mean CV ROC-AUC:  {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # 7. Save Model
    joblib.dump(grid_search.best_estimator_, 'churn_model.pkl')
    joblib.dump(feature_names, 'feature_names.pkl')
    print("\n💾 Model saved to 'churn_model.pkl'")

    # 8. Final Verdict
    print("\n" + "=" * 60)
    print("🏆 FINAL VERDICT")
    print("=" * 60)
    if test_log_loss < 0.50:
        print("✅ EXCELLENT: Model is production-ready!")
    elif test_log_loss < 0.60:
        print("✅ GOOD: Model is usable with monitoring.")
    else:
        print("⚠️ NEEDS IMPROVEMENT: Consider more features or different algorithms.")

    print(f"🎯 Final Test Log Loss: {test_log_loss:.4f}")
    print(f"🎯 Final Test ROC AUC:  {test_auc:.4f}")


if __name__ == "__main__":
    main()