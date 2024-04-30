from sklearn.metrics import accuracy_score, classification_report
from flask import Blueprint, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer
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

nafld1_data = pd.read_csv('nafld.csv')
nafld1_data = nafld1_data.dropna()
data = nafld1_data

X = data.drop(["id", "status", "bmi"], axis=1)
y = data["status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

rf_clf = RandomForestClassifier(max_features=4, n_estimators =100 ,bootstrap = True)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)
# print(classification_report(y_pred, y_test))

# class_names = ['Class 0', 'Class 1']
# explainer = LimeTabularExplainer(X_train.values, class_names=class_names, feature_names=X_train.columns)

class_names = ['No NAFLD', 'Yes NAFLD']
explainer = LimeTabularExplainer(X_train.values, 
                                 class_names=class_names, 
                                 feature_names=X_train.columns)

genai.configure(api_key="AIzaSyADyoNUYW54pe9rZVmDnVriwBtHtX8vrPk")

nafld = Blueprint('nafld', __name__)

@nafld.route('/')
def home():
    return render_template('home.html')

@nafld.route('/nafld')
def nafld_page():
    return render_template('nafld.html')

@nafld.route('/lime_nafld')
def lime():
    return render_template('lime_nafld.html')

@nafld.route('/upload_image_nafld', methods=['POST'])
def upload_pdf():
    if 'imageFile' not in request.files:
        return 'No file part', 400
    
    image_file = request.files['imageFile']
    image_name = str(request.files['imageFile'].filename)
    image_file.save(image_name)

    csv_output = analyse(image_name)
    to_csv(csv_output)

    return 'Image uploaded successfully', 200

@nafld.route('/user_input_nafld')
def get_user_input():
    user_input = pd.read_csv('new_nafld.csv')
    return user_input.to_html()

@nafld.route('/explain_nafld')
def explain():
    # user_input = pd.read_csv('new_nafld.csv')
    # instance = user_input.drop(columns='Outcome').iloc[len(user_input) - 1]
    # explanation = explainer.explain_instance(instance.values, rf_clf.predict_proba)

    read = pd.read_csv('new_nafld.csv')
    instance = read.drop(columns=['id', 'status', 'bmi']).iloc[len(read) - 1].values
    explanation = explainer.explain_instance(instance, rf_clf.predict_proba, num_features=len(X_train.columns))

    html_content = explanation.as_html(show_table=True, show_all=False)

    img_file = 'website/static/nafld.png'
    imgkit.from_string(html_content, img_file)

    user_input_dict = read.to_dict(orient='records')[0]
    return jsonify(user_input=user_input_dict, explanation_image=img_file)

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

def analyse(filename):
    promt = "I'm using Gemini API for image scanning. I need values from the image in CSV format. I need them in this CSV order: id ,age ,male ,weight ,height ,bmi ,case.id ,futime ,status. And status 1 or 0. In male, it is gender, if it is male value is 1 and if it is female values is 0. If some of these values are not present replace their value with 0. I just need the CSV values, I don't need anything else. I'm saying again your response should be only CSV numerical values nothing more"
    model = genai.GenerativeModel('gemini-pro-vision')
    img = PIL.Image.open(filename)
    response = model.generate_content([promt, img], stream=True)
    response.resolve()
    return response.text

def to_csv(text):
    csv_string = text
    rows = [csv_string.split(",")]
    filename = "new_nafld.csv"
    try:
        with open(filename, "a", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(rows)
        print("CSV data has been written to new_nafld.csv")
    except Exception as e:
        print(f"Error occurred while writing to CSV file: {e}")

print("NAFLD ran successfully")