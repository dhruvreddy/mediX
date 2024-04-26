from flask import Blueprint, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import accuracy_score
from IPython.display import Markdown
import google.generativeai as genai
from IPython.display import display
import pandas as pd
import numpy as np
import PIL.Image
import textwrap
import pathlib
import imgkit
import json
import csv

data_URL = "diabetes.csv"
diabetes_data = pd.read_csv(data_URL)
#diabetes_data.drop(['Pregnancies','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction'], axis = 1, inplace = True)

X = diabetes_data.drop(columns='Outcome')
y = diabetes_data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, 
                                                    stratify =y, 
                                                    random_state = 13)
rf_clf = RandomForestClassifier(max_features=2, n_estimators =100 ,bootstrap = True)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

from lime.lime_tabular import LimeTabularExplainer
class_names = ['No diabetes', 'Has diabetes']
feature_names = list(X_train.columns)
explainer = LimeTabularExplainer(X_train.values, feature_names = feature_names, 
                                 class_names = class_names, mode = 'classification')

genai.configure(api_key="AIzaSyADyoNUYW54pe9rZVmDnVriwBtHtX8vrPk")

routes = Blueprint('routes', __name__)

@routes.route('/')
def home():
    return render_template('index.html')

@routes.route('/lime')
def lime():
    return render_template('lime.html')

@routes.route('/upload-image', methods=['POST'])
def upload_pdf():
    if 'imageFile' not in request.files:
        return 'No file part', 400
    
    image_file = request.files['imageFile']
    image_name = str(request.files['imageFile'].filename)
    image_file.save(image_name)

    csv_output = analyse(image_name)
    to_csv(csv_output)

    return 'Image uploaded successfully', 200

@routes.route('/user_input')
def get_user_input():
    user_input = pd.read_csv('user.csv')
    return user_input.to_html()

@routes.route('/explain')
def explain():
    user_input = pd.read_csv('new_user.csv')
    instance = user_input.drop(columns='Outcome').iloc[len(user_input) - 1]
    explanation = explainer.explain_instance(instance.values, rf_clf.predict_proba)

    print("User Input Values:")
    for feature, value in user_input.items():
        print(f"{feature}: {value}")

    # Display model prediction and explanation
    print("\nModel Prediction and Explanation:")
    print(f"Predicted class: {explanation.predicted_class}")
    print(f"Probability: {explanation.predict_proba[explanation.predicted_class]}")
    print("\nTop Features contributing to the prediction:")
    for feature, weight in explanation.as_list():
        print(f"{feature}: {weight}")

    html_content = explanation.as_html(show_table=True, show_all=False)

    img_file = 'website/static/lime_explanation.png'
    imgkit.from_string(html_content, img_file)

    user_input_dict = user_input.to_dict(orient='records')[0]
    return jsonify(user_input=user_input_dict, explanation_image=img_file)

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

def analyse(filename):
    promt = "I'm using Gemini API for image scanning. I need values from the image in CSV format. I need them in this CSV order: Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome. And outcome 1 or 0. If some of these values are not present replace their value with 0. I just need the CSV values, I don't need anything else. I'm saying again your response should be only CSV values nothing more"
    model = genai.GenerativeModel('gemini-pro-vision')
    img = PIL.Image.open(filename)
    response = model.generate_content([promt, img], stream=True)
    response.resolve()
    return response.text

def to_csv(text):
    csv_string = text
    rows = [csv_string.split(",")]
    filename = "new_user.csv"
    try:
        with open(filename, "a", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(rows)
        print("CSV data has been written to user.csv")
    except Exception as e:
        print(f"Error occurred while writing to CSV file: {e}")

