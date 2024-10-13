from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Step 1: Train the model using only 5 features
def train_model():
    # Load the breast cancer dataset
    data = load_breast_cancer()

    # Select only the first 5 features: mean radius, mean texture, mean perimeter, mean area, mean smoothness
    X = data.data[:, :5]
    y = data.target

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest Classifier
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Save the trained model to a file
    with open('breast_cancer_model.pkl', 'wb') as model_file:
        pickle.dump(clf, model_file)

# Call the train_model function to train and save the model
train_model()

# Step 2: Load the saved model
with open('breast_cancer_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Step 3: Define routes for the web app
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data (the 5 features from the user)
    mean_radius = float(request.form['mean_radius'])
    mean_texture = float(request.form['mean_texture'])
    mean_perimeter = float(request.form['mean_perimeter'])
    mean_area = float(request.form['mean_area'])
    mean_smoothness = float(request.form['mean_smoothness'])

    # Create a numpy array for the input features
    input_features = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]])

    # Use the loaded model to predict the result
    prediction = model.predict(input_features)[0]
    
    # Convert prediction to human-readable result
    result = 'Malignant' if prediction == 0 else 'Benign'

    # Render the result in the result.html page
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
