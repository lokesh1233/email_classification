import models.train_model
import pandas as pd


trained_model = model_train_MNB()

def transform_Text(text):
    textFeaturesTest = pd.Series([text])
    textFeaturesTest = textFeaturesTest.apply(clean)
    featuresTest = vectorizer.transform(textFeaturesTest)

