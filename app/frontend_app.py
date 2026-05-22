import streamlit as st
import requests
import pandas as pd
from typing import Optional
import time

# Page config
st.set_page_config(
    page_title="F1-PitStop Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .prediction-box-pit {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .prediction-box-no-pit {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# API configuration
API_URL = st.secrets.get("API_URL", "https://f1-pitstop-ealp.onrender.com")

# Header
st.title("🏎️ F1-PitStop Strategy Predictor")
st.markdown("**Predict optimal pit stop timing for Formula 1 races**")

# Sidebar for API configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_url_input = st.text_input(
        "API URL",
        value=API_URL,
        help="URL of the FastAPI backend"
    )
    
    # Health check
    try:
        response = requests.get(f"{api_url_input}/health", timeout=2)
        if response.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Error")
    except Exception as e:
        st.error(f"❌ Cannot connect to API: {str(e)}")
    
    st.markdown("---")
    st.markdown("""
    ### About
    This tool uses machine learning to predict whether a driver should pit 
    on the next lap based on current race conditions and tire status.
    
    **Features considered:**
    - Driver performance history
    - Tire compound and wear
    - Current position and pace
    - Race progress
    """)


# Main content - Two column layout
col1, col2 = st.columns([1.5, 1], gap="large")

with col1:
    st.header("📊 Input Race Data")
    
    # Create form
    with st.form("prediction_form"):
        # Driver and Race info
        st.subheader("Race Information")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            driver = st.selectbox(
                "Driver",
                ["VER", "HAM", "ALB", "SAI", "LEC", "NOR", "PIA", "CAR", "MAG", "STR"],
                help="Driver abbreviation"
            )
        with col_b:
            year = st.number_input(
                "Year",
                min_value=2015,
                max_value=2024,
                value=2024,
                step=1
            )
        with col_c:
            race = st.selectbox(
                "Race",
                [
                    "Monaco Grand Prix",
                    "British Grand Prix",
                    "Abu Dhabi Grand Prix",
                    "Australian Grand Prix",
                    "Hungarian Grand Prix",
                    "Singapore Grand Prix",
                    "Monza",
                    "Spa",
                    "Silverstone",
                    "Suzuka"
                ]
            )
        
        # Tire and pit stop info
        st.subheader("Tire & Pit Stop Status")
        col_d, col_e, col_f = st.columns(3)
        with col_d:
            compound = st.selectbox(
                "Tire Compound",
                ["SOFT", "MEDIUM", "HARD"],
                help="Current tire compound"
            )
        with col_e:
            tyre_life = st.number_input(
                "Tire Life (laps)",
                min_value=0.0,
                max_value=50.0,
                value=15.0,
                step=0.5
            )
        with col_f:
            pitstop_num = st.number_input(
                "Pit Stop Number",
                min_value=0,
                max_value=3,
                value=1,
                step=1
            )
        
        # Lap and position info
        st.subheader("Current Lap & Position")
        col_g, col_h, col_i = st.columns(3)
        with col_g:
            lap_number = st.number_input(
                "Lap Number",
                min_value=1,
                max_value=100,
                value=25,
                step=1
            )
        with col_h:
            stint = st.number_input(
                "Stint",
                min_value=1,
                max_value=5,
                value=1,
                step=1
            )
        with col_i:
            position = st.number_input(
                "Current Position",
                min_value=1,
                max_value=20,
                value=1,
                step=1
            )
        
        # Performance metrics
        st.subheader("Performance Metrics")
        col_j, col_k, col_l = st.columns(3)
        with col_j:
            lap_time = st.number_input(
                "Lap Time (seconds)",
                min_value=50.0,
                max_value=150.0,
                value=85.5,
                step=0.1
            )
        with col_k:
            lap_time_delta = st.number_input(
                "Lap Time Delta",
                min_value=-10.0,
                max_value=10.0,
                value=-0.5,
                step=0.1,
                help="Difference vs best lap"
            )
        with col_l:
            race_progress = st.number_input(
                "Race Progress",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.05,
                help="0.0 = start, 1.0 = finish"
            )
        
        # Advanced metrics
        st.subheader("Advanced Metrics")
        col_m, col_n = st.columns(2)
        with col_m:
            cum_degradation = st.number_input(
                "Cumulative Degradation",
                min_value=-200.0,
                max_value=0.0,
                value=-10.5,
                step=1.0
            )
        with col_n:
            position_change = st.number_input(
                "Position Change",
                min_value=-20.0,
                max_value=20.0,
                value=0.0,
                step=0.5
            )
        
        # Submit button
        submit_button = st.form_submit_button(
            "🔮 Predict Pit Strategy",
            use_container_width=True,
            type="primary"
        )

with col2:
    st.header("🎯 Prediction Result")
    
    if submit_button:
        # Show spinner while loading
        with st.spinner("Analyzing race data..."):
            try:
                # Prepare request data
                payload = {
                    "Driver": driver,
                    "Compound": compound,
                    "Race": race,
                    "Year": year,
                    "PitStop": pitstop_num,
                    "LapNumber": lap_number,
                    "Stint": stint,
                    "TyreLife": tyre_life,
                    "Position": position,
                    "LapTime_s": lap_time,
                    "LapTime_Delta": lap_time_delta,
                    "Cumulative_Degradation": cum_degradation,
                    "RaceProgress": race_progress,
                    "Position_Change": position_change
                }
                
                # Make API request
                response = requests.post(
                    f"{api_url_input}/predict",
                    json=payload,
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display prediction
                    prediction = result["prediction"]
                    label = result["prediction_label"]
                    confidence = result["confidence_message"]
                    
                    if prediction == 1:
                        st.markdown(
                            f"""
                            <div class="prediction-box-pit">
                            ⚠️ {label} ⚠️<br>
                            <small style="font-size: 0.7em; opacity: 0.9;">{confidence}</small>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        st.success("Strategy: ENTER PIT LANE")
                    else:
                        st.markdown(
                            f"""
                            <div class="prediction-box-no-pit">
                            ✅ {label} ✅<br>
                            <small style="font-size: 0.7em; opacity: 0.9;">{confidence}</small>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        st.info("Strategy: CONTINUE ON TRACK")
                    
                    # Show input summary
                    st.markdown("---")
                    st.subheader("📋 Input Summary")
                    summary = result["input_summary"]
                    col_summary1, col_summary2, col_summary3 = st.columns(3)
                    with col_summary1:
                        st.write("Driver", summary["driver"])
                    with col_summary2:
                        st.write("Year", summary["year"])
                    with col_summary3:
                        st.write("Position", summary["position"])
                    
                    col_summary4, col_summary5 = st.columns(2)
                    with col_summary4:
                        st.write("Race", summary["race"])
                    with col_summary5:
                        st.write("Lap", summary["lap_number"])
                
                else:
                    st.error(f"❌ Prediction failed: {response.text}")
            
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure the backend is running at: " + api_url_input)
            except requests.exceptions.Timeout:
                st.error("❌ Request timeout. The API took too long to respond.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8em;">
    <p>🏎️ F1-PitStop Predictor v1.0 | Built with FastAPI + Streamlit | 🚀</p>
    <p>Disclaimer: This is a demonstration tool. Always follow official pit stop procedures and team strategies.</p>
</div>
""", unsafe_allow_html=True)