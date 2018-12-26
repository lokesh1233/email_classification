import pandas as pd
from bson.json_util import dumps
import json
import spacy
from nltk.stem import SnowballStemmer

from models.train_model import Train_Model
from models.spacy_train import spacy_train_model

class Predict_Model:
    def __init__(self):
        self.train_model_data = Train_Model()
        self.transformData = self.train_model_data.transformData
        
        # changeing model to trained model
        spacy_mdl_dta = spacy_train_model()
        self.nlp = spacy_mdl_dta.trainedModel #spacy.load("en_core_web_lg")
        self.predict_Location = spacy_mdl_dta.predict_Location
        self.predict_Date = spacy_mdl_dta.predict_Date
        self.entities = {"GPE": "Location",
                    "ORG": "organization",
                    "PERSON": "NAME",
                    "DATE": "DATE",
                    "TIME": "Time",
                    "CARDINAL": "CARDINAL",
                    "FAC":"LOCATION"
                    }
        self.stemmer = SnowballStemmer("english").stem
        print()
    
    def tarnsform_raw_text(self):
        textFeaturesTest = pd.Series([self.raw_text])
        textFeaturesTest = textFeaturesTest.apply(self.transformData.clean)
        return self.transformData.vectorizer.transform(textFeaturesTest)

    def text_preprocessing(self, text):
        return ' '.join(text.replace('\n', ' ').replace('1st', '1 st').replace('2nd', '2 nd').replace('3rd', '3 rd').split())
    
    def predict_classification(self, text):
        jsonData = json.loads(text)
        self.raw_text = self.text_preprocessing(jsonData["mail"])
        transformData = self.tarnsform_raw_text()
        pred_data = self.train_model_data.GSlr.predict(transformData)
        entityData = []
        entity = self.nlp(u''+self.raw_text)
        taj_places = self.predict_Location.predict_Location(entity)
        taj_dates = self.predict_Date.date_Predict(entity)
        # removing in prediction of dates
        # PERCENT, QUANTITY, ORDINAL, 
        for token in entity.ents:
            wrd = token.text.strip()
            label_ = token.label_
            if label_ == 'DATE' or label_ == 'CARDINAL':
                continue

            if label_ == 'GPE' or label_ == 'FAC':
                try:
                    plceMatched = False
                    for plce in taj_places:
                        if plce['text'] == token.text.strip():
                            print("matched")
                            plceMatched = True
                            break
                    if plceMatched == True:
                        continue
                    # if taj_places[token.text.strip()]:
                    #     continue
                except:
                    print()

                # continue
            label = self.entities[label_] if label_ in self.entities else label_
            #  and token.text.strip().lower() == taj_places
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
        entes = taj_places + entityData + taj_dates
        # print(entes)
        returnMessage = dumps({"message": pred_data[0], "msgCode": "S", "entity":entes })
        return returnMessage



# text = nlp(sentence)
# entityDtl = []
# for token in text.ents:
#     wrd = token.text.strip()
#     label_ =  token.label_
#     if len(wrd) > 0 and label_ in entities and ((label_ == "PERSON" and wrd.replace(' ','').isalpha() and stemmer(wrd) == wrd.lower()) or label_ != "PERSON"):
#         entityDtl.append({"label":token.label_, "text":token.text.strip()})
