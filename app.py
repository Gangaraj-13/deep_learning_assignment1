import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load model and scaler
model = tf.keras.models.load_model('multi_output_model.h5')
scaler = joblib.load('scaler.pkl')

# Number of features
NUM_FEATURES = 164  # Replace with actual number of features used during training

def preprocess_input(input_data):
    """ Preprocess input data for the model """
    input_data = np.array(input_data).reshape(1, NUM_FEATURES)
    return scaler.transform(input_data)

def predict(input_data):
    """ Make predictions using the model """
    preprocessed_data = preprocess_input(input_data)
    management_pred, severity_pred, diagnosis_pred = model.predict(preprocessed_data)
    
    # Get the class with the highest probability for each output
    management_class = np.argmax(management_pred, axis=1)
    severity_class = np.argmax(severity_pred, axis=1)
    diagnosis_class = np.argmax(diagnosis_pred, axis=1)
    
    return management_class[0], severity_class[0], diagnosis_class[0]

def main():
    st.title('Pediatric Appendicitis Prediction')
    
    st.sidebar.header('Input Features')
    
    # Collect input for 164 features
    input_features = [st.sidebar.number_input(f'Feature {i + 1}', value=0.0) for i in range(NUM_FEATURES)]
    
    if st.button('Predict'):
        try:
            management_class, severity_class, diagnosis_class = predict(input_features)
            st.write(f"Management Class: {management_class}")
            st.write(f"Severity Class: {severity_class}")
            st.write(f"Diagnosis Class: {diagnosis_class}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
