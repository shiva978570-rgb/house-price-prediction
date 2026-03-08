import streamlit as st
import pickle
import numpy as np

# --- PAGE SETUP ---
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

st.title("🏠 Real Estate Price Predictor")
st.write("Apne sapno ke ghar ki details bhariye aur estimated price jaaniye.")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    with open('house_model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    data = load_model()
    model = data['model']
    locations = data['locations']
    loc_map = data['location_mapping']
except:
    st.error("Pehle 'train_house.py' run karein!")
    st.stop()

# --- INPUTS ---
col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("Area / Location", locations)
    sqft = st.number_input("Total Square Feet Area", min_value=100, max_value=10000, value=1200)

with col2:
    bhk = st.selectbox("BHK (Number of Bedrooms)", [1, 2, 3, 4, 5])
    bath = st.selectbox("Number of Bathrooms", [1, 2, 3, 4])

# --- PREDICTION ---
if st.button("Calculate Property Value", use_container_width=True):
    loc_encoded = loc_map[location]
    input_features = [[loc_encoded, sqft, bhk, bath]]
    
    prediction = model.predict(input_features)[0]
    
    st.success(f"### Estimated Market Price: ₹{round(prediction, 2)}")
    st.balloons()