# Cyberbullying Detection

## Problem Statement
Cyberbullying is a growing problem on social media platforms like Twitter.
The volume of content posted every second makes manual moderation impossible.
The goal of this project is to build a machine learning model that automatically
classifies tweets into cyberbullying categories based on their text content.

## Objective
- Perform exploratory data analysis on real-world tweet data
- Clean and preprocess raw tweet text using NLP techniques
- Build and compare multiple text classification models
- Select and deploy the best performing model as a web application

## Dataset
The dataset contains more than 47,000 tweets with 2 columns: text and label.
The target variable is `label` which contains 6 categories:
not_cyberbullying, ethnicity, gender, religion, age, and other_cyberbullying.

## Approach
1. Data Loading & Overview
2. Data Cleaning & EDA (missing values, duplicates, class distribution, tweet length)
3. Text Preprocessing (lowercasing, removing URLs, mentions, hashtags, punctuation)
4. Feature Extraction using TF-IDF Vectorizer
5. Model Training & Evaluation (Logistic Regression, Random Forest, Naive Bayes)
6. Model Comparison (best accuracy: Logistic Regression at 81.07%)
7. Save Best Model & Deploy with Streamlit

## Project Summary & Conclusions

### What We Did
- Loaded a Twitter dataset with more than 47,000 tweets and 2 columns (text, label).
- Checked for null values and duplicates — removed them to ensure clean data.
- Analyzed class distribution, tweet lengths, and category patterns during EDA.
- Cleaned raw tweet text by removing URLs, mentions, hashtags, and punctuation.
- Converted text into numerical features using TF-IDF Vectorizer (5,000 features).
- Trained 3 classification models individually and compared their accuracy.
- Deployed the best model as a live web application using Streamlit.

### What We Observed During EDA
- Dataset contained 6 categories — not_cyberbullying, ethnicity, gender, religion, age, other_cyberbullying.
- Classes were approximately balanced across all categories.
- No significant missing values were found in the dataset.
- Some duplicate tweets were detected and removed before training.
- Cyberbullying tweets tended to be slightly longer than non-cyberbullying tweets.
- Ethnicity and gender categories contained the most aggressive language patterns.

### Model Results
| Model | Accuracy |
|---|---|
| Logistic Regression | 81.07% |
| Random Forest | 79.21% |
| Naive Bayes | 75.43% |

### Why Logistic Regression Performed Best
- Logistic Regression works very well with high-dimensional sparse data like TF-IDF vectors.
- It learns a weight for each word which makes it effective at separating text categories.
- It is less likely to overfit on text data compared to tree-based models.
- Random Forest struggled because TF-IDF produces thousands of features which slows it down and reduces its advantage.

### Why Naive Bayes Performed Worst
- Naive Bayes assumes all words are independent of each other which is rarely true in real text.
- It ignores word order and context which limits its understanding of language patterns.
- Despite this it is still a fast and reasonable baseline for text classification.

### Key Takeaways
- Text preprocessing was a critical step — noisy tweets hurt model performance significantly.
- TF-IDF is a strong and simple baseline for converting text into features.
- Logistic Regression with 81.07% accuracy is the best model for this task.
- The model can help social media platforms automatically flag harmful content at scale.

### Future Improvements
- We can try deep learning models like LSTM or BERT for better contextual understanding.
- We can add stopword removal and lemmatization to improve text preprocessing.
- We can handle any class imbalance using SMOTE if needed.
- We can expand the dataset with more recent tweets to improve generalization.