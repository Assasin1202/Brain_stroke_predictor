import streamlit as st
import pandas as pd
import numpy as np
import onnxruntime as ort
import lime.lime_tabular
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score
import plotly.graph_objects as go
import plotly.express as px
import requests
import os

# Set the page configuration with custom theme
st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
    <style>
    /* Main background and text colors */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Header styling */
    .main .block-container {
        padding-top: 2rem;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        border-radius: 5px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
    }
    
    /* Input fields styling */
    .stSelectbox, .stNumberInput {
        background-color: #262730;
        border-radius: 5px;
        padding: 1rem;
    }
    
    /* Text input styling */
    .stTextInput>div>div>input {
        background-color: #262730;
        color: #FAFAFA;
    }
    
    /* Selectbox styling */
    .stSelectbox>div>div>div {
        background-color: #262730;
        color: #FAFAFA;
    }
    
    /* Number input styling */
    .stNumberInput>div>div>input {
        background-color: #262730;
        color: #FAFAFA;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #262730;
        color: #FAFAFA;
        border-radius: 5px;
        padding: 1rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #262730;
        border-radius: 5px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #FAFAFA;
    }
    
    /* Custom div styling */
    .custom-div {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    /* Risk level divs */
    .risk-high {
        background-color: #3D1F1F;
        color: #FF4B4B;
    }
    
    .risk-low {
        background-color: #1F3D1F;
        color: #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Header with logo and title
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🫀 Stroke Risk Predictor</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #FAFAFA;'>Predict your stroke risk using advanced machine learning</p>", unsafe_allow_html=True)
    
    # Load and preprocess data
    @st.cache_data
    def load_data():
        df = pd.read_csv("resampled-stroke-data.csv")
        df['bmi'].fillna(df['bmi'].median(), inplace=True)
        return df

    df = load_data()
    X = df.drop(columns=["stroke"])
    y = df["stroke"]
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    continuous_features = ["age", "avg_glucose_level", "bmi"]
    scaler = StandardScaler()
    X_train[continuous_features] = scaler.fit_transform(X_train[continuous_features])
    X_test[continuous_features] = scaler.transform(X_test[continuous_features])

    # Load ONNX model
    @st.cache_resource
    def load_onnx_model():
        # Use the quantized model as per user change
        model_path = "big_mlp_model_quant.onnx"
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}. Please ensure it's in the correct directory.")
            return None, None
        try:
            sess = ort.InferenceSession(model_path)
            return sess, sess.get_inputs()[0].name
        except Exception as e:
            st.error(f"Error loading ONNX model: {e}")
            return None, None

    sess, input_name = load_onnx_model()
    if sess is None:
        st.stop() # Stop execution if model loading failed

    def onnx_predict_proba(X):
        X_np = np.array(X, dtype=np.float32)
        preds = sess.run(None, {input_name: X_np})[0]
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        if preds.shape[1] == 1:
            preds = np.hstack([1 - preds, preds])
        return preds

    # Create explainers
    @st.cache_resource
    def create_explainers(X_train):
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns.tolist(),
            class_names=["No Stroke", "Stroke"],
            mode='classification',
            discretize_continuous=False
        )
        background_sample = X_train.sample(50, random_state=42)
        shap_explainer = shap.KernelExplainer(model=onnx_predict_proba, data=background_sample)
        return lime_explainer, shap_explainer

    lime_explainer, shap_explainer = create_explainers(X_train)

    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Model Performance", "ℹ️ About"])

    with tab1:
        st.markdown("<h2 style='color: #FF4B4B;'>Make a Prediction</h2>", unsafe_allow_html=True)
        
        # Create three columns for better layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<h3 style='color: #FAFAFA;'>Personal Information</h3>", unsafe_allow_html=True)
            age = st.number_input("Age", min_value=1, max_value=120, value=50, step=1)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
            
        with col2:
            st.markdown("<h3 style='color: #FAFAFA;'>Health Metrics</h3>", unsafe_allow_html=True)
            avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0, step=1.0)
            hypertension = st.selectbox("Hypertension", ["No", "Yes"])
            heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
            
        with col3:
            st.markdown("<h3 style='color: #FAFAFA;'>Lifestyle Factors</h3>", unsafe_allow_html=True)
            smoking_status = st.selectbox("Smoking Status", 
                ["Never Smoked", "Formerly Smoked", "Smokes"])
            work_type = st.selectbox("Work Type", 
                ["Private", "Self-employed", "Government", "Never Worked", "Children"])
            residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
            ever_married = st.selectbox("Ever Married", ["No", "Yes"])

        # Convert inputs to model format
        input_data = {
            "age": age,
            "hypertension": 1 if hypertension == "Yes" else 0,
            "heart_disease": 1 if heart_disease == "Yes" else 0,
            "avg_glucose_level": avg_glucose_level,
            "bmi": bmi,
            "gender_Male": 1 if gender == "Male" else 0,
            "gender_Other": 1 if gender == "Other" else 0,
            "ever_married_Yes": 1 if ever_married == "Yes" else 0,
            "work_type_Never_worked": 1 if work_type == "Never Worked" else 0,
            "work_type_Private": 1 if work_type == "Private" else 0,
            "work_type_Self-employed": 1 if work_type == "Self-employed" else 0,
            "work_type_children": 1 if work_type == "Children" else 0,
            "Residence_type_Urban": 1 if residence_type == "Urban" else 0,
            "smoking_status_formerly smoked": 1 if smoking_status == "Formerly Smoked" else 0,
            "smoking_status_never smoked": 1 if smoking_status == "Never Smoked" else 0,
            "smoking_status_smokes": 1 if smoking_status == "Smokes" else 0
        }
        
        input_df = pd.DataFrame([input_data])
        input_df_scaled = input_df.copy()
        input_df_scaled[continuous_features] = scaler.transform(input_df_scaled[continuous_features])

        # Center the predict button
        col1_btn, col2_btn, col3_btn = st.columns([1,2,1])
        with col2_btn:
            if st.button("🔍 Predict Stroke Risk"):
                sample_array = input_df_scaled.values
                pred_proba = onnx_predict_proba(sample_array)
                pred_label = np.argmax(pred_proba, axis=1)[0]
                stroke_prob = pred_proba[0][1] * 100

                # Create a gauge chart for risk visualization with dark theme
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = stroke_prob,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Stroke Risk Probability", 'font': {'color': "#FAFAFA"}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickcolor': "#FAFAFA"},
                        'bar': {'color': "#FF4B4B"},
                        'steps': [
                            {'range': [0, 30], 'color': "#1F3D1F"},
                            {'range': [30, 70], 'color': "#3D3D1F"},
                            {'range': [70, 100], 'color': "#3D1F1F"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': stroke_prob
                        }
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': "#FAFAFA"}
                )
                
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Display risk level with custom styling
                risk_level_text = "High Risk" if pred_label == 1 else "Low Risk"
                risk_div_class = "risk-high" if pred_label == 1 else "risk-low"
                st.markdown(f"""
                    <div class='custom-div {risk_div_class}' style='text-align: center;'>
                        <h3>⚠️ {risk_level_text} of Stroke</h3>
                        <p style='font-size: 24px;'>{stroke_prob:.1f}% probability</p>
                    </div>
                """, unsafe_allow_html=True)

                # Display explanations in expandable sections with better styling
                with st.expander("📊 View LIME Explanation", expanded=False):
                    lime_exp = lime_explainer.explain_instance(
                        data_row=sample_array[0],
                        predict_fn=onnx_predict_proba,
                        num_features=8,
                        labels=(0, 1)
                    )
                    
                    # Create a bar chart for LIME values with dark theme
                    lime_values = lime_exp.as_list(label=1)
                    fig = px.bar(
                        x=[abs(v[1]) for v in lime_values],
                        y=[v[0] for v in lime_values],
                        orientation='h',
                        title='Feature Importance (LIME)',
                        labels={'x': 'Impact on Prediction', 'y': 'Feature'}
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': "#FAFAFA"},
                        xaxis={'gridcolor': '#262730'},
                        yaxis={'gridcolor': '#262730'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with st.expander("📈 View SHAP Explanation", expanded=False):
                    shap_values_local = shap_explainer.shap_values(input_df_scaled)
                    
                    # Handle different return formats from SHAP explainer
                    # If shap_values_local is a list with two elements (binary classification standard format)
                    if isinstance(shap_values_local, list) and len(shap_values_local) > 1:
                        # Use the positive class (index 1) SHAP values
                        local_shap = shap_values_local[1][0]
                    else:
                        # If it's a single array, use it directly
                        # This handles the case where shap returns a single array of values
                        local_shap = shap_values_local[0] if isinstance(shap_values_local, list) else shap_values_local
                    
                    shap_explanation = sorted(
                        [(col, input_df_scaled[col].values[0], local_shap[idx]) 
                         for idx, col in enumerate(input_df_scaled.columns)],
                        key=lambda x: abs(x[2]),
                        reverse=True
                    )
                    
                    # Create a bar chart for SHAP values with dark theme
                    fig = px.bar(
                        x=[v[2] for v in shap_explanation],
                        y=[v[0] for v in shap_explanation],
                        orientation='h',
                        title='Feature Importance (SHAP)',
                        labels={'x': 'SHAP Value', 'y': 'Feature'}
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': "#FAFAFA"},
                        xaxis={'gridcolor': '#262730'},
                        yaxis={'gridcolor': '#262730'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # --- OpenAI Integration --- 
                # --- Hugging Face Integration --- 
                st.markdown("<hr style='border-top: 1px solid #262730;'>", unsafe_allow_html=True)
                st.markdown("<h3 style='color: #FAFAFA;'>🤖 AI Summary & Insights (via Hugging Face)</h3>", unsafe_allow_html=True)
                
                # Get Hugging Face API Token
                hf_token = st.secrets.get("HUGGINGFACE_API_TOKEN")
                if not hf_token:
                    hf_token = st.text_input("Enter your Hugging Face API Token to get the summary:", type="password")
                
                if hf_token:
                    try:
                        # Define the Hugging Face Inference API endpoint
                        # Using Mistral-7B-Instruct as an example
                        model_id = "mistralai/Mistral-7B-Instruct-v0.1"
                        api_url = f"https://api-inference.huggingface.co/models/{model_id}"
                        headers = {"Authorization": f"Bearer {hf_token}"}
                        
                        # Prepare data for the prompt (same as before)
                        input_details = "\n".join([f"- {key.replace('_', ' ').title()}: {value}" for key, value in input_data.items()])
                        
                        # Safely format LIME factors - make sure lime_values exists and has expected structure
                        try:
                            top_lime_factors = "\n".join([f"- {feat}: {weight:.3f} (LIME impact)" for feat, weight in lime_values[:5]])
                        except (NameError, TypeError, IndexError):
                            top_lime_factors = "LIME explanation not available"
                        
                        # Safely format SHAP factors - make sure shap_explanation exists and has expected structure
                        try:
                            top_shap_factors = "\n".join([f"- {feat}: {shap_val:.3f} (SHAP value)" for feat, _, shap_val in shap_explanation[:5]])
                        except (NameError, TypeError, IndexError):
                            top_shap_factors = "SHAP explanation not available"

                        # Construct the prompt for the instruction-following model
                        # Note: System prompts aren't standard in the basic Inference API like in OpenAI's Chat API
                        prompt = f"""
[INST] You are a helpful assistant explaining health prediction results clearly and empathetically. Analyze the following stroke risk prediction based on patient data and model explanations: 

Patient Data:
{input_details}

Prediction Result:
- Risk Level: {risk_level_text}
- Probability of Stroke: {stroke_prob:.1f}%

Key Factors Identified by Model Explainers:
LIME Factors (showing impact towards stroke risk):
{top_lime_factors}
SHAP Factors (showing contribution to prediction):
{top_shap_factors}

Instructions:
Provide a concise summary in simple, natural language for the patient. Address the following points:
1. Briefly explain the overall risk level and probability.
2. Based *only* on the provided data and factors, explain the key reasons *why* the model reached this conclusion (mentioning 1-2 most significant factors).
3. Suggest 1-2 general lifestyle areas (like diet, exercise, smoking, etc.) that the patient might focus on for potential risk reduction, *if applicable* based on their inputs (e.g., if BMI is high, suggest diet/exercise; if smoker, suggest cessation). Frame these as general health suggestions, *not* specific medical advice.
4. Keep the tone empathetic, clear, and easy to understand for someone without a medical background.
5. Do NOT give definitive medical advice or make diagnoses. [/INST]
"""
                        
                        # Prepare the payload for the Inference API
                        payload = {
                            "inputs": prompt,
                            "parameters": {
                                "max_new_tokens": 250, # Similar to max_tokens
                                "temperature": 0.6,   # Adjust temperature as needed
                                "return_full_text": False # We only want the generated part
                            }
                        }
                        
                        with st.spinner("Generating AI summary via Hugging Face..."):
                            response = requests.post(api_url, headers=headers, json=payload)
                            
                            if response.status_code == 200:
                                result = response.json()
                                # The response format might vary slightly; adjust based on actual API output
                                # Often it's a list containing a dict with 'generated_text'
                                if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
                                    summary = result[0]['generated_text'].strip()
                                    st.markdown(f"<div class='custom-div'>{summary}</div>", unsafe_allow_html=True)
                                else:
                                    st.error(f"Unexpected response format from Hugging Face API: {result}")
                            elif response.status_code == 401:
                                st.error("Authentication failed. Please check your Hugging Face API Token.")
                            else:
                                st.error(f"Error from Hugging Face API: {response.status_code} - {response.text}")
                                
                    except requests.exceptions.RequestException as e:
                         st.error(f"Network error connecting to Hugging Face API: {e}")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                else:
                    st.warning("Hugging Face API Token not provided. Cannot generate summary.")

    with tab2:
        st.markdown("<h2 style='color: #FF4B4B;'>Model Performance</h2>", unsafe_allow_html=True)
        X_test_np = X_test.astype(np.float32).values
        pred_proba_test = onnx_predict_proba(X_test_np)
        y_pred_test = np.argmax(pred_proba_test, axis=1)
        cm = confusion_matrix(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)
        
        # Create a more visually appealing confusion matrix with dark theme
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted No Stroke', 'Predicted Stroke'],
            y=['Actual No Stroke', 'Actual Stroke'],
            colorscale='Reds'
        ))
        fig.update_layout(
            title='Confusion Matrix',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#FAFAFA"},
            xaxis={'gridcolor': '#262730'},
            yaxis={'gridcolor': '#262730'}
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown(f"""
                <div class='custom-div' style='text-align: center;'>
                    <h3 style='color: #FF4B4B;'>Model Metrics</h3>
                    <p style='font-size: 24px; color: #FAFAFA;'>F1 Score: {f1:.3f}</p>
                </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.markdown("<h2 style='color: #FF4B4B;'>About This App</h2>", unsafe_allow_html=True)
        st.markdown("""
            <div class='custom-div'>
                <h3 style='color: #FAFAFA;'>🔬 Overview</h3>
                <p style='color: #FAFAFA;'>This application uses advanced machine learning to predict stroke risk based on various health and demographic factors. 
                The model has been trained on historical patient data and provides detailed explanations for its predictions.</p>
                
                <h3 style='color: #FAFAFA;'>🎯 Features</h3>
                <ul style='color: #FAFAFA;'>
                    <li>Real-time stroke risk prediction</li>
                    <li>Interactive input interface</li>
                    <li>Detailed explanation of predictions using LIME and SHAP</li>
                    <li>Visual performance metrics</li>
                </ul>
                
                <h3 style='color: #FAFAFA;'>⚡ How to Use</h3>
                <ol style='color: #FAFAFA;'>
                    <li>Enter your personal and health information</li>
                    <li>Click the "Predict Stroke Risk" button</li>
                    <li>View your risk assessment and detailed explanations</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
