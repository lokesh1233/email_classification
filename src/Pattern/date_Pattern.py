import json

# Load required modules
# import pandas as pd
# import numpy as np

from spacy.matcher import Matcher
from spacy.attrs import IS_PUNCT, LOWER


class date_Pattern:
    
    def __init__(self, mdl):
        
        self.nlp = mdl
        pattern_Data = self.generate_Patterns(self.load_Data())
        self.matcher = Matcher(self.nlp.vocab)
        self.date_matcher = self.buildMatcher(pattern_Data)


    def load_Data(self):
        
        # open and read the file data
        with open('C:/Users/Lokesh/jupyter works/email_classification/src/Pattern/date_Pattern.json', 'r') as jsonFile:
            jsonData = json.load(jsonFile)
        return jsonData['date_pattern']
    
    def generate_Patterns(self, dateArr):
        
        # creating patterns for the date formats
        date_pattern_val = []
        for dte in dateArr:
            date_pattern_val.append((dte['example'][0],  dte['pattern']))
        return date_pattern_val

    def on_match(self, matcher, doc, id, matches):
         #   print(matcher, doc, id, matches)
        return matches

    def buildMatcher(self, patterns):
        for pattern in patterns:
            self.matcher.add(pattern[0], self.on_match, pattern[1])
        return self.matcher
    
    def date_Matcher(self, matcher, doc):
        skills = []
#         doc = nlp(text.lower())
        matches = matcher(doc)
        mat_data = []
        for b in matches: 
            match_id, start, end = b
            mat_data.append({"text":doc[start : end].text , "label":"DATE"})
        #         print(doc[start : end], start, end)
        return mat_data

    def date_Predict(self, doc):
        return self.date_Matcher(self.date_matcher, doc)
    