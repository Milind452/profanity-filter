from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
from data_preprocessing import load_and_preprocess_data

# Load and preprocess data
X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess_data()

# Initialize and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred, target_names=['Hate Speech', 'Offensive Language', 'Neither']))

# Save the model and vectorizer
joblib.dump(model, 'profanity_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
