from flask import Flask, render_template, request
import pickle
import joblib

app = Flask(__name__)
model = joblib.load('model.aiml')
model2 = joblib.load('model2.aiml')

@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    gre_score = int(request.form['GRE Score'])
    toefl_score = int(request.form['TOEFL Score'])
    university_rating = int(request.form['University Rating'])
    sop = int(request.form['SOP'])
    lor = int(request.form['LOR'])
    cgpa = int(request.form['CGPA'])
    research = int(request.form['Research'])

    sampletest = [[gre_score, toefl_score, university_rating, sop, lor, cgpa, research]]
    prediction = model.predict(sampletest)
    percentage = prediction * 100
    output = round(percentage[0], 2)

    return render_template('index.html', prediction_message=f'The chance of admission is: {output}%')

#classification
@app.route("/index2")
def hello2():
    return render_template('index2.html')

@app.route("/predict2", methods=['POST'])
def predict2():
    answer = request.form.to_dict()
    # print(answer)
    fixedAcidity = float(answer['fixedAcidity'])
    volatileAcidity = float(answer['volatileAcidity'])
    citricAcidity = float(answer['Citric Acidity'])
    residualSugar = float(answer['Residual Sugar'])
    chlorides = float(answer['Chlorides'])
    freeSulfurDioxide = float(answer['Free Sulfur Dioxide'])
    totalSulfurDioxide = float(answer['Total Sulfur Dioxide'])
    density = float(answer['Density'])
    ph = float(answer['ph'])
    sulphates = float(answer['Sulphates'])
    alcohol = float(answer['Alcohol'])

    sampletest = [[fixedAcidity, volatileAcidity, citricAcidity, residualSugar, chlorides, freeSulfurDioxide, totalSulfurDioxide, density, ph, sulphates, alcohol]]
    prediction = model2.predict(sampletest)

    if prediction[0] == 1:
        result = "Good quality wine"
    else :
        result = "Bad quality wine"

    return render_template('index2.html', prediction_message=f'Result: {result}')


if __name__ == "__main__":
    app.run(debug=True)
