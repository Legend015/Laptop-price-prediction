import streamlit as st
import joblib
import numpy as np

# Import the model and data
pipe = joblib.load('pipe.pkl')
df = joblib.load('df.pkl')

st.title("Laptop Price Predictor")

# Brand
company = st.selectbox('Brand', df['Company'].unique())

# Type of laptop
type_laptop = st.selectbox('Type', df['TypeName'].unique())

# RAM
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight
weight = st.number_input('Weight of the Laptop', min_value=0.0)

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# Screen size
screen_size = st.slider('Screen size (in inches)', 10.0, 18.0, 13.0)

# Resolution
resolution = st.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
    '2880x1800', '2560x1600', '2560x1440', '2304x1440'
])

# CPU
cpu = st.selectbox('CPU', df['Cpu brand'].unique())

# Storage: HDD and SSD
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox('GPU', df['Gpu brand'].unique())

# Operating System
os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    try:
        # Process user inputs
        touchscreen = 1 if touchscreen == 'Yes' else 0
        ips = 1 if ips == 'Yes' else 0
        X_res, Y_res = map(int, resolution.split('x'))
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

        # Prepare query for prediction
        query = np.array([company, type_laptop, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os], dtype=object)
        query = query.reshape(1, -1)

        # Debugging output
        print("Query shape:", query.shape)
        print("Query data:", query)
        print("Query dtypes:", [type(x) for x in query[0]])

        # Make prediction
        predicted_price = int(np.exp(pipe.predict(query)[0]))
        st.title(f"The predicted price of this configuration is: {predicted_price}Rs.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
