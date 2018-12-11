from flask import Flask, render_template, request
# import os
# print(os.getcwd())

from models.predict_model import Predict_Model

predictModel = Predict_Model()

app=Flask(__name__,  template_folder='template')
@app.route("/")
def home():
    return render_template("home.html")


#Email classification post data    
@app.route('/EmailClassifySet',methods = ['POST'])
def EmailClassify():
    if request.method == 'POST':
        return predictModel.predict_classification(request.data)
    else:
        return ''

    
    
if __name__ =="__main__":
    app.run(debug=True)