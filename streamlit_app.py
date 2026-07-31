import streamlit as st
import requests
import json

# Set page configuration
st.set_page_config(
    page_title="Credit Risk Analyzer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #0068c9;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0051a8;
        border: none;
    }
    .prediction-card {
        padding: 2rem;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .approve {
        color: #28a745;
        font-weight: bold;
        font-size: 2rem;
    }
    .reject {
        color: #dc3545;
        font-weight: bold;
        font-size: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("💳 Credit Risk Assessment Dashboard")
st.markdown("Enter applicant details to analyze the credit risk.")
st.markdown("---")

# Define available models (dynamic fetch with fallback)
default_models = ["decision_tree", "gradient_boosting", "knn", "logistic_l1", "logistic_l2", "logistic", "random_forest", "svm"]
available_models = default_models

try:
    health_resp = requests.get("http://localhost:8000/", timeout=2)
    if health_resp.status_code == 200:
        api_models = health_resp.json().get("models_available", [])
        if api_models:
            available_models = sorted(list(set(default_models + api_models)))
except:
    pass

# Sidebar for inputs and global controls
with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox("Select Model", available_models, index=available_models.index("logistic") if "logistic" in available_models else 0)
    
    st.markdown("---")
    st.header("Applicant Information")
    st.info("💡 **WoE Hint**: Lower values typically indicate higher risk of default. Try negative values to see 'Reject' decisions.")
    
    credit_utilization = st.number_input("Credit Utilization Ratio WoE", value=0.0, step=0.1)
    debt_to_income = st.number_input("Debt to Income Ratio WoE", value=0.0, step=0.1)
    annual_income = st.number_input("Annual Income WoE", value=0.0, step=0.1)
    employment_years = st.number_input("Employment Years WoE", value=0.0, step=0.1)
    late_payments = st.number_input("Late Payments (12m) WoE", value=0.0, step=0.1)
    
    st.markdown("---")
    threshold = st.slider("Acceptance Threshold (PD)", 0.0, 1.0, 0.5, 0.01)
    st.caption(f"Applications with PD > {threshold:.2f} will be rejected.")

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Prediction Analysis ({selected_model})")
    st.write("Click the button below to analyze the credit risk based on the provided features.")
    
    if st.button("Predict Credit Risk"):
        payload = {
            "credit_utilization_ratio_woe": credit_utilization,
            "debt_to_income_ratio_woe": debt_to_income,
            "employment_years_woe": employment_years,
            "number_of_late_payments_12m_woe": late_payments,
            "annual_income_woe": annual_income
        }
        
        try:
            # API endpoint with dynamic threshold and model selection
            api_url = f"http://localhost:8000/predict?threshold={threshold}&model_name={selected_model}"
            response = requests.post(api_url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                pd = result.get("PD", 0.0)
                decision = result.get("decision", "Unknown")
                
                # Display result in a nice card
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>Risk Analysis Result</h3>
                    <p style="font-size: 1.2rem;">Probability of Default (PD)</p>
                    <p style="font-size: 2.5rem; font-weight: bold;">{pd:.2%}</p>
                    <hr>
                    <p style="font-size: 1.2rem;">Final Decision (via {selected_model})</p>
                    <p class="{'approve' if decision == 'Approve' else 'reject'}">{decision.upper()}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Progress bar for visual risk levels
                st.write("Risk Level Visualization")
                st.progress(pd)
                
            else:
                st.error(f"Error: API returned status code {response.status_code}")
                st.write(response.json().get("error", "Unknown error"))
        except Exception as e:
            st.error(f"Could not connect to the API. Make sure the FastAPI server is running at http://localhost:8000")

with col2:
    st.subheader("Feature Guide")
    st.markdown("""
    - **WoE (Weight of Evidence)**: A transformation technique used in credit scoring.
    - **PD (Probability of Default)**: Likelihood of failing to make payments.
    - **Threshold**: Applications with high PD are rejected.
    """)
    
    st.info("Note: Negative WoE values typically indicate higher risk for certain features.")

st.markdown("---")
st.caption("Credit Risk Project - Multi-Model Assessment Tool")
