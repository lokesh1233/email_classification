import pandas as pd
from bson.json_util import dumps
import json
import spacy
from nltk.stem import SnowballStemmer

from models.train_model import Train_Model

class Predict_Model:
    def __init__(self):
        self.train_model_data = Train_Model()
        self.transformData = self.train_model_data.transformData
        self.nlp = spacy.load("en_core_web_lg")
        self.entities = {"GPE": "Location",
                    "ORG": "organization",
                    "PERSON": "Name",
                    "DATE": "Date",
                    "TIME": "Time",
                    "CARDINAL": "CARDINAL"
                    }
        self.stemmer = SnowballStemmer("english").stem
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
            wrd = token.text.strip()
            label_ = token.label_
            label = self.entities[label_] if label_ in self.entities else label_
            if len(wrd) > 0:
                entityData.append({"label":label.upper(), "text":token.text.strip()})

            #     if len(wrd) > 0 and label_ in entities and ((label_ == "PERSON" and wrd.replace(' ','').isalpha() and stemmer(wrd) == wrd.lower()) or label_ != "PERSON"):
            #         entityDtl.append({"label":token.label_, "text":token.text.strip()})
            # if len(wrd) > 0 and ((label_ == "PERSON" and wrd.replace(' ', '').isalpha() and self.stemmer(
            #         wrd) == wrd.lower()) or label_ != "PERSON"):
            #     entityData.append({"label": label.upper(), "text": wrd})
            # if len(wrd) > 0 and label_ in self.entities and ((label_ == "PERSON" and wrd.replace(' ',
            #                                                                                 '').isalpha() and self.stemmer(
            #         wrd) == wrd.lower()) or label_ != "PERSON"):
            #     entityData.append({"label": self.entities[label_].upper(), "text": token.text.strip()})

            # if len(token.text.rstrip()) > 0:
            #     entityData.append({"text": token.text, "label": token.label_})

        returnMessage = dumps({"message": pred_data[0], "msgCode": "S", "entity":entityData})
        return returnMessage



# text = nlp(sentence)
# entityDtl = []
# for token in text.ents:
#     wrd = token.text.strip()
#     label_ =  token.label_
#     if len(wrd) > 0 and label_ in entities and ((label_ == "PERSON" and wrd.replace(' ','').isalpha() and stemmer(wrd) == wrd.lower()) or label_ != "PERSON"):
#         entityDtl.append({"label":token.label_, "text":token.text.strip()})
