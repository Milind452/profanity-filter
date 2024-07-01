import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_and_preprocess_data(filename='data.csv'):
    data = pd.read_csv(filename)
    data = data[['text', 'label']]  # Ensure these columns exist
    data['text'] = data['text'].astype(str).str.lower()

    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    return X_train_vect, X_test_vect, y_train, y_test, vectorizer
