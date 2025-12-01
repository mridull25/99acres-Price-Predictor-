import streamlit as st
import pandas as pd
from scraper import fetch_data
from model import PricePredictor
import plotly.express as px

st.set_page_config(page_title="99acres Price Predictor", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
    }
    .css-1d391kg {
        padding: 2rem;
        border-radius: 10px;
        background: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1f2937;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏠 99acres Property Price Predictor")
st.markdown("<p style='text-align: center; color: #6b7280;'>Scrape property data and predict market value using Machine Learning.</p>", unsafe_allow_html=True)

# Sidebar for controls
st.sidebar.header("Configuration")
location_query = st.sidebar.text_input("Enter Location to Scrape", "Sector 62")
scrape_btn = st.sidebar.button("Scrape & Train Model")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = PricePredictor()
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Scraping and Training Logic
if scrape_btn:
    with st.spinner(f"Scraping 2000+ records for {location_query}..."):
        df = fetch_data(location_query)
        st.session_state.data = df
        st.success(f"Successfully fetched {len(df)} records!")
        
    with st.spinner("Training ML Model..."):
        metrics = st.session_state.model.train(df)
        st.session_state.model_trained = True
        st.success(f"Model Trained! MAE: ₹{metrics['mae']:,.2f}, R2 Score: {metrics['r2']:.2f}")

# Main Content Area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Property Data")
    if st.session_state.data is not None:
        st.dataframe(st.session_state.data)
        
        # Visualization
        fig = px.scatter(st.session_state.data, x="Area_SqFt", y="Price", color="BHK", 
                         title="Price vs Area (Colored by BHK)", hover_data=['Age_Years', 'Location'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please scrape data to view insights.")

with col2:
    st.subheader("💰 Predict Price")
    
    if st.session_state.model_trained:
        with st.form("prediction_form"):
            locations = [
                'Gurugram Sector 14', 'Dilshad Garden', 'Shahdara', 
                'Gurugram Sector 56', 'Gurugram Sector 45',
                'Sector 62 Noida', 'Indirapuram'
            ]
            loc = st.selectbox("Location", locations)
            
            # Display estimated market rate
            base_rates = {
                'Dilshad Garden': 9000,
                'Shahdara': 7500,
                'Gurugram Sector 14': 12000,
                'Gurugram Sector 56': 11000,
                'Gurugram Sector 45': 13500,
                'Sector 62 Noida': 7500,
                'Indirapuram': 6000
            }
            st.info(f"Estimated Market Rate for {loc}: ₹{base_rates.get(loc, 5000)}/sq.ft")
            
            col_a, col_b = st.columns(2)
            with col_a:
                area = st.number_input("Area (Sq. Ft.)", min_value=500, max_value=10000, value=1200)
                age = st.slider("Property Age (Years)", 0, 50, 5)
            with col_b:
                bhk = st.selectbox("BHK", [1, 2, 3, 4, 5], index=1)
                # Use key to force update when location changes
                market_rate = st.number_input("Market Rate (₹/Sq. Ft.)", min_value=1000, value=base_rates.get(loc, 5000), key=f"rate_{loc}")
            
            with st.expander("Advanced Location Factors"):
                col_c, col_d = st.columns(2)
                with col_c:
                    transit = st.slider("Proximity to Transit (km)", 0.1, 10.0, 2.0, help="Distance to nearest Metro/Train/Bus Hub")
                    school = st.slider("School District Rating (1-10)", 1.0, 10.0, 7.0, help="Quality of local schools")
                    walkability = st.slider("Walkability Score (0-100)", 0, 100, 70, help="Convenience of the neighborhood")
                with col_d:
                    green_space = st.slider("Green Space Access (0-100%)", 0.0, 1.0, 0.3, help="Percentage of area covered by parks/water")
                    crime = st.slider("Crime Rate Index (0-100)", 0.0, 100.0, 30.0, help="Lower is better")

            st.markdown("### Select Amenities")
            amenities_options = ['Lift', 'Security', 'Park', 'Gym', 'Power Backup', 'Parking']
            selected_amenities = st.multiselect("Amenities", amenities_options, default=['Lift', 'Parking'])
            
            predict_btn = st.form_submit_button("Predict Price")
            
            if predict_btn:
                input_data = {
                    'Location': loc,
                    'Area_SqFt': area,
                    'Age_Years': age,
                    'BHK': bhk,
                    'Market_Rate_SqFt': market_rate,
                    'Proximity_to_Transit_km': transit,
                    'School_Rating': school,
                    'Walkability_Score': walkability,
                    'Green_Space_Area': green_space,
                    'Crime_Rate_Index': crime
                }
                
                # Add amenities to input
                for amenity in amenities_options:
                    input_data[amenity] = 1 if amenity in selected_amenities else 0
                
                price = st.session_state.model.predict(input_data)
                st.balloons()
                st.metric(label="Estimated Price", value=f"₹{price:,.2f}")
    else:
        st.warning("Please train the model first by clicking 'Scrape & Train Model' in the sidebar.")
