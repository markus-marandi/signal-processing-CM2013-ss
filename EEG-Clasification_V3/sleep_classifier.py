import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
import pickle
import os
import time

# Define output directory and create it if it doesn't exist
output_dir = os.path.join(os.getcwd(), '@Output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# Helper function to join path with output directory
def get_output_path(filename):
    return os.path.join(output_dir, filename)

# Check if CSV file exists
input_csv_path = get_output_path('eeg_features.csv')
if not os.path.exists(input_csv_path):
    # Check if it exists in the current directory as fallback
    if os.path.exists('eeg_features.csv'):
        input_csv_path = 'eeg_features.csv'
    else:
        print("Error: eeg_features.csv file not found. Please run the main.py script first.")
        exit(1)

print("\nBuilding sleep stage classification models...")

# Load the feature DataFrame
feature_df = pd.read_csv(input_csv_path)

# Remove any NaN values
feature_df = feature_df.dropna()

# Remove rows where sleep_stage = 1 (empty label) or any invalid stage
valid_stages = [0, 2, 3, 4, 5]  # REM, N3, N2, N1, Wake
feature_df = feature_df[feature_df['sleep_stage'].isin(valid_stages)]

# Define X (features) and y (target labels)
X = feature_df.drop(['timestamp', 'sleep_stage'], axis=1)
y = feature_df['sleep_stage']

# Print the number of samples for each sleep stage
print("Sleep stage distribution:")
stage_counts = y.value_counts().sort_index()
stage_mapping = {0: 'REM', 2: 'N3', 3: 'N2', 4: 'N1', 5: 'Wake'}
for stage, count in stage_counts.items():
    stage_name = stage_mapping.get(stage, f'Unknown {stage}')
    print(f"  {stage_name}: {count} samples")

# Use a LabelEncoder to transform non-consecutive labels to consecutive integers (0, 1, 2, 3, 4)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Store the original class names in the same order as the encoded labels
encoded_classes = [stage_mapping[stage] for stage in label_encoder.classes_]

# Create training and validation sets with 40:60 ratio
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.6, random_state=42, stratify=y_encoded)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")

# First, let's check for feature importance using a preliminary Random Forest
preliminary_rf = RandomForestClassifier(n_estimators=100, random_state=42)
preliminary_rf.fit(X_train, y_train)

# Get feature importances
importances = preliminary_rf.feature_importances_
feature_names = X.columns

# Plot feature importances
plt.figure(figsize=(12, 8))
indices = np.argsort(importances)[::-1]
plt.bar(range(len(indices[:20])), importances[indices[:20]], align='center')
plt.xticks(range(len(indices[:20])), [feature_names[i] for i in indices[:20]], rotation=90)
plt.title('Top 20 Feature Importances')
plt.tight_layout()
plt.savefig(get_output_path('feature_importance.png'))
plt.show()

# Select the top 20 features for models to prevent overfitting
selector = SelectFromModel(preliminary_rf, threshold=-np.inf, max_features=20)
selector.fit(X_train, y_train)

X_train_selected = selector.transform(X_train)
X_val_selected = selector.transform(X_val)

# Get names of selected features
selected_feature_indices = selector.get_support(indices=True)
selected_feature_names = [feature_names[i] for i in selected_feature_indices]
print("\nTop 20 selected features:")
for i, feature in enumerate(selected_feature_names):
    print(f"{i+1}. {feature}")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_val_scaled = scaler.transform(X_val_selected)

# Perform hyperparameter tuning via cross-validation
# This will help improve performance on all classes, especially the N1 stage
print("\nPerforming hyperparameter tuning...")

# 1. Random Forest Classifier
print("\nTuning Random Forest classifier...")
start_time = time.time()

# Define parameter grid
rf_param_grid = {
    'n_estimators': [100, 200], 
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

# Use GridSearchCV with stratified k-fold
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42), 
    rf_param_grid, 
    cv=5, 
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=1
)

rf_grid.fit(X_train_scaled, y_train)
print(f"Best Random Forest parameters: {rf_grid.best_params_}")
print(f"Time taken: {time.time() - start_time:.2f} seconds")

# Train the best model
rf_model = rf_grid.best_estimator_
rf_pred = rf_model.predict(X_val_scaled)
rf_accuracy = accuracy_score(y_val, rf_pred)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# 2. Support Vector Machine
print("\nTuning SVM classifier...")
start_time = time.time()

# Define parameter grid
svm_param_grid = {
    'C': [1, 5, 10],
    'gamma': ['scale', 'auto', 0.1],
    'class_weight': ['balanced', None]
}

# Use GridSearchCV with stratified k-fold
svm_grid = GridSearchCV(
    SVC(kernel='rbf', probability=True, random_state=42), 
    svm_param_grid, 
    cv=5, 
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=1
)

svm_grid.fit(X_train_scaled, y_train)
print(f"Best SVM parameters: {svm_grid.best_params_}")
print(f"Time taken: {time.time() - start_time:.2f} seconds")

# Train the best model
svm_model = svm_grid.best_estimator_
svm_pred = svm_model.predict(X_val_scaled)
svm_accuracy = accuracy_score(y_val, svm_pred)
print(f"SVM Accuracy: {svm_accuracy:.4f}")

# 3. XGBoost
print("\nTuning XGBoost classifier...")
start_time = time.time()

# Define parameter grid
xgb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [5, 7, 9],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'scale_pos_weight': [1, 3, 5]  # To help with class imbalance
}

# Create XGBoost classifier
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(encoded_classes),
    random_state=42,
    n_jobs=-1
)

# Use GridSearchCV with stratified k-fold
xgb_grid = GridSearchCV(
    xgb_model, 
    xgb_param_grid, 
    cv=5, 
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=1
)

xgb_grid.fit(X_train_scaled, y_train)
print(f"Best XGBoost parameters: {xgb_grid.best_params_}")
print(f"Time taken: {time.time() - start_time:.2f} seconds")

# Train the best model
xgb_model = xgb_grid.best_estimator_
xgb_pred = xgb_model.predict(X_val_scaled)
xgb_accuracy = accuracy_score(y_val, xgb_pred)
print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")

# Compare model accuracies
model_accuracies = {
    'Random Forest': rf_accuracy,
    'SVM': svm_accuracy,
    'XGBoost': xgb_accuracy
}

# Find the best model
best_model_name = max(model_accuracies, key=model_accuracies.get)
best_model_accuracy = model_accuracies[best_model_name]
print(f"\nBest model: {best_model_name} with accuracy {best_model_accuracy:.4f}")

# Detailed classification report for all models
print("\n--- Random Forest Classification Report ---")
print(classification_report(y_val, rf_pred, target_names=encoded_classes))

print("\n--- SVM Classification Report ---")
print(classification_report(y_val, svm_pred, target_names=encoded_classes))

print("\n--- XGBoost Classification Report ---")
print(classification_report(y_val, xgb_pred, target_names=encoded_classes))

# Plot confusion matrices
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=encoded_classes,
               yticklabels=encoded_classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(get_output_path(f'{title.replace(" ", "_").lower()}.png'))
    plt.show()

# Plot confusion matrices for all models
plot_confusion_matrix(y_val, rf_pred, 'Random Forest Confusion Matrix')
plot_confusion_matrix(y_val, svm_pred, 'SVM Confusion Matrix')
plot_confusion_matrix(y_val, xgb_pred, 'XGBoost Confusion Matrix')

# Bar plot comparing model accuracies
plt.figure(figsize=(10, 6))
plt.bar(model_accuracies.keys(), model_accuracies.values())
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for i, (model, acc) in enumerate(model_accuracies.items()):
    plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
plt.tight_layout()
plt.savefig(get_output_path('model_comparison.png'))
plt.show()

# Save the best model
best_model = None
if best_model_name == 'Random Forest':
    best_model = rf_model
elif best_model_name == 'SVM':
    best_model = svm_model
else:  # XGBoost
    best_model = xgb_model

with open(get_output_path('best_sleep_classifier.pkl'), 'wb') as f:
    pickle.dump({
        'model': best_model,
        'scaler': scaler,
        'selector': selector,
        'feature_names': feature_names,
        'selected_features': selected_feature_names
    }, f)

print(f"\nBest model saved as '@Output/best_sleep_classifier.pkl'")

# Generate learning curves to evaluate if more data would help
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=5,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    return plt

# Plot learning curve for the best model
print(f"\nGenerating learning curve for {best_model_name}...")

# Use a smaller subset for learning curve to speed up computation
X_subset = X_train_scaled[:min(1000, len(X_train_scaled))]
y_subset = y_train[:min(1000, len(y_train))]

if best_model_name == 'Random Forest':
    lc_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
elif best_model_name == 'SVM':
    lc_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
else:  # XGBoost
    lc_model = xgb.XGBClassifier(n_estimators=50, learning_rate=0.1, max_depth=7, random_state=42)
    
plot_learning_curve(lc_model, f'Learning Curve ({best_model_name})', 
                   X_subset, y_subset[:len(X_subset)], ylim=(0.5, 1.01), cv=5, n_jobs=-1)
plt.savefig(get_output_path('learning_curve.png'))
plt.show()

print("\nClassification analysis complete!") 