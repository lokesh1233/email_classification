import pandas as pd
from bson.json_util import dumps
import json
import spacy

from models.train_model import Train_Model

class Predict_Model:
    def __init__(self):
        self.train_model_data = Train_Model()
        self.transformData = self.train_model_data.transformData
        self.nlp = spacy.load("en_core_web_sm")
        print()
    
    def tarnsform_raw_text(self):
        textFeaturesTest = pd.Series([self.raw_text])
        textFeaturesTest = textFeaturesTest.apply(self.transformData.clean)
        return self.transformData.vectorizer.transform(textFeaturesTest)
    
    
    def predict_classification(self, text):
        jsonData = json.loads(text)
        self.raw_text = jsonData["mail"]
        transformData = self.tarnsform_raw_text()
        pred_data = self.train_model_data.mnb.predict(transformData)
        entityData = []
        entity = self.nlp(self.raw_text)
        for token in entity.ents:
            if len(token.text.rstrip()) > 0:
                entityData.append({"text": token.text, "label": token.label_})

        returnMessage = dumps({"message": pred_data[0], "msgCode": "S", "entity":entityData})
        return returnMessage

