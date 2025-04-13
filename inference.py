import pandas as pd
import numpy as np
import onnxruntime as ort
import lime.lime_tabular
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score
import warnings
import sys
import random

warnings.filterwarnings("ignore")

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------
data_path = "resampled-stroke-data.csv"
df = pd.read_csv(data_path)

# Fill missing BMI values if necessary
if df['bmi'].isnull().sum() > 0:
    df['bmi'].fillna(df['bmi'].median(), inplace=True)

# Separate features and target
X = df.drop(columns=["stroke"])
y = df["stroke"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale continuous features
continuous_features = ["age", "avg_glucose_level", "bmi"]
scaler = StandardScaler()
X_train[continuous_features] = scaler.fit_transform(X_train[continuous_features])
X_test[continuous_features] = scaler.transform(X_test[continuous_features])

# -------------------------------
# ONNX Model Prediction (MLP)
# -------------------------------
onnx_model_path = "big_mlp_model.onnx"
sess = ort.InferenceSession(onnx_model_path)
input_name = sess.get_inputs()[0].name

def onnx_predict_proba(X):
    """
    Accepts input data X and returns predicted class probabilities.
    If the output is one-dimensional or has only one column, it reshapes it to create
    two columns (binary classification: [1-p, p]).
    """
    X_np = np.array(X, dtype=np.float32)
    preds = sess.run(None, {input_name: X_np})[0]
    # Ensure output is 2-dimensional
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    # If only one probability is returned per sample, compute the two-class probabilities
    if preds.shape[1] == 1:
        preds = np.hstack([1 - preds, preds])
    return preds

# Get predictions for test set using the ONNX model.
X_test_np = X_test.astype(np.float32).values
pred_proba = onnx_predict_proba(X_test_np)
# Convert probabilities to class labels (assumes positive class is at index 1)
y_pred = np.argmax(pred_proba, axis=1)

print("ONNX Model Predictions:")
print(y_pred)

# -------------------------------
# Evaluation Metrics
# -------------------------------
cm = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
print("\nF1 Score:")
print(f1)
sys.stdout.flush()  # Flush outputs to ensure they print before any progress bars

# -------------------------------
# LIME and SHAP Explanations
# -------------------------------
# Create a LIME explainer using training data
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=["No Stroke", "Stroke"],
    mode='classification',
    discretize_continuous=False
)

# Create a SHAP KernelExplainer with a background sample from the training data.
background_sample = X_train.sample(50, random_state=42)
shap_explainer = shap.KernelExplainer(model=onnx_predict_proba, data=background_sample)

def explain_sample(instance_index):
    print(f"\n=== Test Sample (Index {instance_index}) ===")
    # Print the original (unscaled) data from the dataframe.
    print(df.loc[instance_index])
    
    # Prepare the instance data (scaled) for explanation
    instance_array = X_test.loc[instance_index].values
    
    # ---- LIME Explanation ----
    lime_exp = lime_explainer.explain_instance(
        data_row=instance_array,
        predict_fn=onnx_predict_proba,
        num_features=8,
        labels=(0, 1)
    )
    print("\n--- LIME Explanation for 'Stroke' ---")
    for feature, weight in lime_exp.as_list(label=1):
        print(f"Feature: {feature:<40} Weight: {weight:.4f}")
    
    # ---- SHAP Explanation ----
    shap_values_local = shap_explainer.shap_values(X_test.loc[[instance_index]])
    local_shap = shap_values_local[1][0]
    print("\n--- SHAP Explanation for 'Stroke' ---")
    for idx in np.argsort(np.abs(local_shap))[::-1]:
        feat_name = X_test.columns[idx]
        feat_value = X_test.loc[instance_index, feat_name]
        shap_val = local_shap[idx]
        print(f"Feature: {feat_name:<20} Value: {feat_value:.3f}  SHAP: {shap_val:.4f}")

# -------------------------------
# Explain a Random Test Sample
# -------------------------------
sample_index = random.choice(X_test.index)
explain_sample(sample_index)
