import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def load_and_preprocess_data(filename='dataset/labeled_data.csv'):
    data = pd.read_csv(filename)
    
    # Filter columns
    data = data[['tweet', 'class']]

    # Convert text to lowercase
    data['tweet'] = data['tweet'].str.lower()

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data['tweet'], data['class'], test_size=0.2, random_state=42)

    # Vectorize text data
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    return X_train_vect, X_test_vect, y_train, y_test, vectorizer
