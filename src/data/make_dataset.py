# -*- coding: utf-8 -*-
# import click
# import logging
# from pathlib import Path
# from dotenv import find_dotenv, load_dotenv

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import string
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split

#sklearn
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
# def main(input_filepath, output_filepath):
#     """ Runs data processing scripts to turn raw data from (../raw) into
#         cleaned data ready to be analyzed (saved in ../processed).
#     """
#     logger = logging.getLogger(__name__)
#     logger.info('making final data set from raw data')

class DataTransform:
     def __init__(self):
        self.appendFilesData = []
        data = pd.DataFrame(self.tarnsforming_raw_text())
        self.features_train = self.vector_transform(data)
        self.labels_train = data['class_to_exec']
     
     def clean(self, doc):
        
        words_to_exclude = set(stopwords.words('english'))
        exclude = set(string.punctuation)
        
        # removing singular and plural nouns 18_12 and added stemming
        tagged_sentence = nltk.tag.pos_tag(doc.split())
        edited_sentence = ' '.join([word for word,tag in tagged_sentence if tag != 'NNP' and tag != 'NNPS'])

        word_free = " ".join([i for i in edited_sentence.lower().split() if i not in words_to_exclude])
        punc_free = ''.join(ch for ch in word_free if ch not in exclude)
        alpha_free = " ".join(word for word in punc_free.split() if word.isalpha())
        stemmer = SnowballStemmer("english").stem
        stem_free = " ".join(stemmer(stem) for stem in alpha_free.split())
        lemma = WordNetLemmatizer()
        normalized = " ".join(lemma.lemmatize(word) for word in stem_free.split())

        return normalized
    

     def tarnsforming_raw_text(self):
        # Import the email modules we'll need
        import glob
        import email
        import mailparser
        from email import policy
        from email.parser import BytesParser

        path = 'C:/Users/Lokesh/jupyter works/email_classification/datawe/raw/Email_Classification/*'
        email_types = glob.glob(path)
        self.appendFilesData = []
        for folder in email_types:
            files = glob.glob(folder + "/*.txt")
            email_type = folder.split('\\')[1]
            for name in files:
                try:
                    with open(name) as fp:
                        raw_data = fp.read()
                        #                     file_raw_data.append(raw_data)
                        msg = mailparser.parse_from_string(raw_data)
                        self.appendFilesData.append({
                            "to": msg.to,
                            "from": msg.from_,
                            "subject": msg.subject,
                            "date": msg.date,
                            #                     "sent":msg["Sent"],
                            #                     "importance":msg["Importance"],
                            "content": raw_data,
                            "class_to_exec": email_type,
                        })

                except IOError as exc:
                    print('Exception')

        return self.appendFilesData


     def vector_transform(self, data):
        textFeatures = data['content'].copy()
        textFeatures = textFeatures.apply(self.clean)
        self.vectorizer = TfidfVectorizer("english", smooth_idf=False, sublinear_tf=False, analyzer='word')
        return self.vectorizer.fit_transform(textFeatures)


#     features_train, features_test, labels_train, labels_test = train_test_split(features, data['class_to_exec'], test_size=0.2, random_state=111)






# if __name__ == '__main__':
#     log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]

#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())

#     main()
