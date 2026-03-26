import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

# OPTIONAL: load test data for accuracy
# (Save these from your notebook if needed)
try:
    X_test = pickle.load(open("X_test.pkl", "rb"))
    y_test = pickle.load(open("y_test.pkl", "rb"))
    accuracy = model.score(X_test, y_test)
except:
    accuracy = None

st.title("🔮 Machine Learning Prediction App")

st.write("Enter feature values to get prediction")

# 👉 CHANGE these inputs based on your dataset
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)

# Convert input into array
input_data = np.array([[feature1, feature2, feature3]])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)

    st.success(f"Prediction: {prediction[0]}")

    # If classification model
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_data)
        st.info(f"Confidence: {np.max(prob)*100:.2f}%")

# Show accuracy if available
if accuracy is not None:
    st.subheader("📊 Model Accuracy")
    st.write(f"Accuracy: {accuracy * 100:.2f}%")
else:
    st.warning("⚠️ Accuracy not available (upload test data)")


