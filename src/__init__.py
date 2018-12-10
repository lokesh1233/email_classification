from flask import Flask, render_template
# import os
# print(os.getcwd())

import data.make_dataset
import models.train_model

trained_model = model_train_MNB()

app=Flask(__name__,  template_folder='template')
@app.route("/")
def home():
    return render_template("home.html")


#Email classification post data    
@app.route('/VRNMaster',methods = ['POST'])
def VRNMasterList():
    if request.method == 'POST':
        return self.VRNHeader.createVRN(request.data)
    else:
        return ''

    
    
if __name__ =="__main__":
    app.run(debug=True)