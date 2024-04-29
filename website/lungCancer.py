from imports import *

data = pd.read_csv("lung_cancer.csv")

X = data.drop(columns='LUNG_CANCER')
y = data['LUNG_CANCER']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, 
                                                    stratify =y, 
                                                    random_state = 13)

rf_clf = RandomForestClassifier(max_features=4, n_estimators =100 ,bootstrap = True)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)
print(classification_report(y_pred, y_test))

class_names = ['No cancer', 'Cancer']
feature_names = list(X_train.columns)
explainer = LimeTabularExplainer(X_train.values, feature_names = feature_names, 
                                 class_names = class_names, mode = 'classification')

lungCancer = Blueprint('lungCancer', __name__)

@lungCancer.route('/lung_cancer')
def index():
    return render_template('lungCancer.html')

@lungCancer.route('lime_lung_cancer')
def lime_lung_cancer():
    return render_template('lime_lung_cancer.html')

@lungCancer.route('/submit', methods=['POST'])
def submit():
    # Get form data and convert 'Yes'/'No' to 1/0
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    smoking = int(request.form['smoking'])
    yellow_fingers = int(request.form['yellow_fingers'])
    anxiety = int(request.form['anxiety'])
    peer_pressure = int(request.form['peer_pressure'])
    chronic_disease = int(request.form['chronic_disease'])
    fatigue = int(request.form['fatigue'])
    allergy = int(request.form['allergy'])
    wheezing = int(request.form['wheezing'])
    alcohol = int(request.form['alcohol'])
    coughing = int(request.form['coughing'])
    breath = int(request.form['breath'])
    swallowing = int(request.form['swallowing'])
    chest_pain = int(request.form['chest_pain'])
    # Add other form fields similarly...

    # Store data in CSV file
    with open('new_lung_cancer.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, allergy, wheezing, alcohol, coughing, breath, swallowing, chest_pain])  # Write data to CSV row

    return 'Data submitted successfully!'

@lungCancer.route('/user_input_lung_cancer')
def get_user_input():
    user_input = pd.read_csv('new_lung_cancer.csv')
    return user_input.to_html()

@lungCancer.route('/explain_lung_cancer')
def explain():
    user_input = pd.read_csv('new_user.csv')
    instance = user_input.drop(columns='Outcome').iloc[len(user_input) - 1]
    explanation = explainer.explain_instance(instance.values, rf_clf.predict_proba)

    html_content = explanation.as_html(show_table=True, show_all=False)

    img_file = 'website/static/lung_cancer.png'
    imgkit.from_string(html_content, img_file)

    user_input_dict = user_input.to_dict(orient='records')[0]
    return jsonify(user_input=user_input_dict, explanation_image=img_file)