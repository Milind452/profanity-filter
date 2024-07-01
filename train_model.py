from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from data_preprocessing import load_and_preprocess_data

X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess_data()

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')

joblib.dump(model, 'profanity_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
