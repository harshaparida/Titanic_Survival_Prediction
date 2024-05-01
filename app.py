from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#app = Flask(__name__)
app = Flask(__name__, static_url_path='/static') 
# Load the dataset
dataset = pd.read_csv('train.csv')

# Data preprocessing
dataset = dataset.drop(columns='Cabin', axis=1)
dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
dataset['Fare'].fillna(dataset['Fare'].mean(), inplace=True)  # Add this line to handle missing Fare values
dataset.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)

# Define features (X) and target (y)
X = dataset.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
y = dataset['Survived']

# Train the model
model = LogisticRegression()
model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = {
        'Pclass': int(request.form['Pclass']),
        'Sex': 1 if request.form['Sex'].lower() == 'female' else 0,
        'Age': float(request.form['Age']),
        'SibSp': int(request.form['SibSp']),
        'Parch': int(request.form['Parch']),
        'Embarked': {'C': 1, 'Q': 2, 'S': 3}.get(request.form['Embarked'].upper(), 3),
        'Fare': float(request.form['Fare'])
    }

    user_df = pd.DataFrame([user_input], columns=X.columns)
    prediction = model.predict(user_df)

    if prediction[0] == 1:
        result = "The passenger is predicted to survive."
    else:
        result = "The passenger is predicted not to survive."
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    port = 8080
    app.run(debug=True, port=port)

