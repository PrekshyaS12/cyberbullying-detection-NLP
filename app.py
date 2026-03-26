import streamlit as st
import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Cyberbullying Detector")

tweet = st.text_area("Enter a tweet:")

if st.button("Predict"):
    cleaned = tweet.lower()
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    st.write("Result:", prediction)