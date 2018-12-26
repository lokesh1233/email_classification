import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import string
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
import sklearn.model_selection as modsel

## trasnforming custom raw data
from data.make_dataset import DataTransform

#
#
# def clean(doc):
#     words_to_exclude = set(stopwords.words('english'))
#     exclude = set(string.punctuation)
#
#     word_free = " ".join([i for i in doc.lower().split() if i not in words_to_exclude])
#     punc_free = ''.join(ch for ch in word_free if ch not in exclude)
#     alpha_free = " ".join(word for word in punc_free.split() if word.isalpha())
#     stemmer = SnowballStemmer("english").stem
#     stem_free = " ".join(stemmer(stem) for stem in alpha_free.split())
#     lemma = WordNetLemmatizer()
#     normalized = " ".join(lemma.lemmatize(word) for word in alpha_free.split())
#
#     return normalized
#
# def tarnsforming_raw_text():
#     # Import the email modules we'll need
#     import glob
#     import email
#     import mailparser
#     from email import policy
#     from email.parser import BytesParser
#
#     path = '../datawe/raw/Email_Classification/*'
#     email_types = glob.glob(path)
#     appendFilesData = []
#     file_raw_data = []
#     for folder in email_types:
#         files = glob.glob(folder+"/*.txt")
#         email_type = folder.split('\\')[1]
#         for name in files:
#             try:
#                 with open(name) as fp:
#                     raw_data = fp.read()
# #                     file_raw_data.append(raw_data)
#                     msg = mailparser.parse_from_string(raw_data)
#                     appendFilesData.append({
#                         "to":msg.to,
#                         "from":msg.from_,
#                         "subject":msg.subject,
#                         "date":msg.date,
#     #                     "sent":msg["Sent"],
#     #                     "importance":msg["Importance"],
#                         "content":msg.body,
#                         "class_to_exec":email_type,
#                     })
#
#             except IOError as exc:
#                 print('Exception')
#
#     return appendFilesData
#
#
# def vector_transform(data):
#     textFeatures = data['content'].copy()
#     textFeatures = textFeatures.apply(clean)
#     vectorizer = TfidfVectorizer("english", smooth_idf=False, sublinear_tf=False, analyzer='word')
#     return vectorizer.fit_transform(textFeatures)
#
# #     features_train, features_test, labels_train, labels_test = train_test_split(features, data['class_to_exec'], test_size=0.2, random_state=111)
#
#

class Train_Model:
    
    def __init__(self):
        self.transformData = DataTransform()
        self.features_train = self.transformData.features_train
        self.labels_train = self.transformData.labels_train
        self.model_train_MNB()
        self.model_train_GridSearch_LR()

# multinominal naive nayes
    def model_train_MNB(self):
        self.mnb = MultinomialNB(alpha=0.2)
        self.mnb.fit(self.features_train, self.labels_train)
    #     prediction = mnb.predict(features_test)
    #     accuracy_score(labels_test,prediction)
    
        return self.mnb

    def model_train_SVM(self):
        self.svc = SVC(kernel='sigmoid', gamma=1.0)
        self.svc.fit(features_train, labels_train)
    #     prediction = mnb.predict(features_test)
    #     accuracy_score(labels_test,prediction)
    
        return self.svc

    def model_train_LR(self):
        self.lr = LogisticRegression()
        self.lr.fit(features_train, labels_train)
    #     prediction = mnb.predict(features_test)
    #     accuracy_score(labels_test,prediction)
    
        return self.lr
    
    def model_train_GridSearch_LR(self):
        ## grid serach CV for tuning hyperparameters
        param_grid_ = {'C': [1e-5, 1e-3, 1e-1, 1e0, 1e1, 1e2]}
        self.GSlr = modsel.GridSearchCV(LogisticRegression(), cv=5, param_grid=param_grid_)
        self.GSlr.fit(self.features_train, self.labels_train)
#         self.GSlr = LogisticRegression()
#         self.GSlr.fit(features_train, labels_train)
    #     prediction = mnb.predict(features_test)
    #     accuracy_score(labels_test,prediction)
    
        return self.GSlr



# if __name__ =="__main__":
#     data = pd.DataFrame(tarnsforming_raw_text())
#     features_train = vector_transform(data)
#     labels_train = data['class_to_exec']