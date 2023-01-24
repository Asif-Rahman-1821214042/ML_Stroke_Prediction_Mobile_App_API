from flask import Flask,request, jsonify
import pickle
import numpy as np

model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)




@app.route('/')
def hello_world():
    return "Hello Patient"


@app.route('/predict', methods=['POST'])
def predict():
    gender=request.form.get('gender')
    age = request.form.get('age')
    hypertension = request.form.get('hypertension')
    heart_disease = request.form.get('heart_disease')
    ever_married = request.form.get('ever_married')
    work_type= request.form.get('work_type')
    Residence_type = request.form.get('Residence_type')
    avg_glucose_level = request.form.get('avg_glucose_level')
    bmi = request.form.get('bmi')

    input_np = np.array([[gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi]])
    prd = model.predict(input_np)[0]
    return jsonify({'result':str(prd)})


if __name__ == '__main__':
    app.run(debug=True)