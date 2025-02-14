import streamlit as st
import pickle
import pandas as pd
import re
import string


# Load the saved models and vectorizer
def load_models():
    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    models = {}
    model_files = [
        "logistic_regression",
        "decision_tree",
        "gradient_boost",
        "random_forest",
    ]

    for model_name in model_files:
        with open(f"models/{model_name}_model.pkl", "rb") as f:
            models[model_name] = pickle.load(f)

    return vectorizer, models


# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub("\[.*?\]", "", text)
    text = re.sub("\\W", " ", text)
    text = re.sub("https?://\S+|www\.\S+", "", text)
    text = re.sub("<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\w*\d\w*", "", text)
    return text


def predict_fake_news(text, vectorizer, models):
    # Preprocess the input text
    processed_text = preprocess_text(text)

    # Transform the text using the loaded vectorizer
    vectorized_text = vectorizer.transform([processed_text])

    # Get predictions from all models
    predictions = {}
    for name, model in models.items():
        pred = model.predict(vectorized_text)[0]
        predictions[name] = "Fake News" if pred == 0 else "Not Fake News"

    return predictions


# Streamlit UI
def main():
    st.title("Fake News Detection System")
    st.write(
        "This application uses multiple machine learning models to detect fake news."
    )

    # Load models
    vectorizer, models = load_models()

    # Text input
    news_text = st.text_area("Enter the news text to analyze:", height=200)

    if st.button("Analyze"):
        if news_text:
            with st.spinner("Analyzing..."):
                # Get predictions
                predictions = predict_fake_news(news_text, vectorizer, models)

                # Display results
                st.subheader("Analysis Results:")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("Model")
                    for model_name in predictions.keys():
                        st.write(model_name.replace("_", " ").title())

                with col2:
                    st.write("Prediction")
                    for prediction in predictions.values():
                        if prediction == "Fake News":
                            st.write("🚫 Fake News")
                        else:
                            st.write("✅ Not Fake News")

                # Calculate consensus
                fake_count = sum(
                    1 for pred in predictions.values() if pred == "Fake News"
                )
                real_count = len(predictions) - fake_count

                st.subheader("Overall Consensus:")
                if fake_count > real_count:
                    st.error("This news appears to be FAKE based on majority voting.")
                else:
                    st.success("This news appears to be REAL based on majority voting.")
        else:
            st.warning("Please enter some text to analyze.")


if __name__ == "__main__":
    main()
