from flask import Flask,render_template,request
app = Flask(__name__)
import pickle

file = open('model.pkl','rb')
clf = pickle.load(file)
file.close()

@app.route('/',methods=["GET","POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        fever = int(myDict['fever'])
        age = int(myDict['age'])
        pain = int(myDict['pain'])
        runnyNose = int(myDict['runnyNose'])
        breath = int(myDict['breathingdifficulty'])
        inputFeatures = [fever,pain,age,runnyNose,breath]
        infProb =clf.predict_proba([inputFeatures])[0][1]
        print(infProb)
        return render_template('show.html',inf = round(infProb*100))
    return render_template('index.html')

if __name__ == "__main__"    :
    app.run(debug = True)