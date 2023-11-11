import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template


# Create flask app
flask_app = Flask(__name__,template_folder="template")
model=pickle.load(open("BrainStroke.p","rb"))
@flask_app.route("/")
def Home():
    return render_template("index.html")

gender={'Male':1 ,'Female':0}
general={'Yes':1 ,'No':0}
worktype={'Private':1, 'Self_employed':2, 'Govt_job':3, 'children':4}
res={'Urban':0 ,'Rural':1}
smoke={'formerly_smoked':1 ,'never_smoked':2 ,'smokes':3, 'Unknown':4}
stroke={0:"Low risk of Brain stroke",1:"High risk of Brain Stroke"}
@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features =[]
    float_features.append(gender[request.form['gender']])
    float_features.append(int(request.form['age']))
    float_features.append(general[request.form['hypertension']])
    float_features.append(general[request.form['heart_disease']])
    float_features.append(general[request.form['ever_married']])
    float_features.append(worktype[request.form['work_type']])
    float_features.append(res[request.form['residence_type']])
    float_features.append(float(request.form['avg_glucose_level']))
    float_features.append(float(request.form['bmi']))
    float_features.append(smoke[request.form['smoking_status']])
    float_features=np.array(float_features)
    float_features=float_features.reshape(1,-1)
    predict=model.predict(float_features)
    result=stroke[predict[0]]
    return render_template("index.html", prediction_text = "You have {}".format(result))

if __name__ == "__main__":
    flask_app.run(debug=True)