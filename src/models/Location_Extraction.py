# Load required modules
import pandas as pd
import numpy as np

from spacy.matcher import Matcher
from spacy.attrs import IS_PUNCT, LOWER
import spacy


class Location_Extraction_model:
    def __init__(self, mdl):
#         nlp = spacy.load('en_core_web_lg')
        self.nlp = mdl
        self.matcher = Matcher(self.nlp.vocab)
        taj_locations = self.load_location()
        # patterns = buildPatterns(taj_locations)
        patterns = self.buildPatterns(taj_locations)
        self.city_matcher = self.buildMatcher(patterns)
#         return self.predict_Location

    def predict_Location(self, doc):
        return self.city_Matcher(self.city_matcher, doc)
    
    def skillPattern(self, skill):
        pattern = []
        for b in skill.split():
            pattern.append({'LOWER':b})  
        return pattern

    def buildPatterns(self, skills):
        pattern = []
        for skill in skills:
            pattern.append(self.skillPattern(skill))
        return list(zip(skills, pattern))
    
    def on_match(self, matcher, doc, id, matches):
        return matches

    def buildMatcher(self, patterns):
        for pattern in patterns:
            self.matcher.add(pattern[0], self.on_match, pattern[1])
        return self.matcher
    
#     def city_Matcher(self, matcher, text, doc):
#         skills = []
# #         doc = nlp(text.lower())
#         matches = matcher(doc)
#         for b in matches:
#             match_id, start, end = b
#             print(doc[start : end], start, end)

    def city_Matcher(self, matcher, doc):
        skills = []
#         doc = nlp(text.lower())
        matches = matcher(doc)
        mat_data = []
        for b in matches:
            match_id, start, end = b
            # mat_data.append(
            mat_data.append({"text":doc[start : end].text , "label":"LOCATION"})
            # print(doc[start : end].text, start, end)
        return [dict(t) for t in {tuple(d.items()) for d in mat_data}]
        # return mat_data

    def load_location(self):

        df = pd.read_csv(r'C:/Users/Lokesh/jupyter works/email_classification/datawe/raw/Email_Classification/Taj Locations.csv')
        taj_cities = []
        self.concatinateLoc(df["city"], taj_cities)
        self.concatinateLoc(df["country"], taj_cities)
        taj_place = []
        for place, repeated in enumerate(df["palace"]):
            taj_place += repeated.lower().split(',')
        taj_cities += [plc.strip() for plc in taj_place]
        return taj_cities 
    
    def concatinateLoc(self, dta, cityArr):
        for city, repeated in dta.value_counts().items():
            cityArr.append(u''+city.lower())
        return cityArr
