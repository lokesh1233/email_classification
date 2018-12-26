import pandas as pd
import numpy as np
import spacy
import random
from spacy.util import minibatch, compounding

#internal class
from models.Location_Extraction import Location_Extraction_model
from Pattern.date_Pattern import date_Pattern

class spacy_train_model:
    def __init__(self):
       # self.nlp = spacy.load('en_core_web_lg')
        df = pd.read_csv("C:/Users/Lokesh/jupyter works/email_classification/datawe/raw/Email_Classification/email_entity_cleansed.csv")
        entityDf = df[df.apply(lambda x: x["text"][x["start_char"]:x["end_char"]] == x["name"], axis=1)]
        entityTrainData = []
        for text, item in  entityDf['text'].value_counts().items():
            mulItems = entityDf[entityDf['text'] == text]
            multipleEntities = []
            for dta in mulItems.values:
                multipleEntities.append((dta[4], dta[1], dta[2]))
            entityTrainData.append((text, {'entities':multipleEntities}))

        self.trainedModel = self.main_train(model = 'en_core_web_lg', TRAIN_DATA = entityTrainData)
        self.predict_Location = Location_Extraction_model(self.trainedModel)
        self.predict_Date = date_Pattern(self.trainedModel)
        
    def remove_whitespace_entities(self, doc):
        doc.ents = [e for e in doc.ents if not e.text.isspace()]
        return doc
    #
    # def text_preprocessing(self, text):
    #     text = text.replace('1st', '1st')
    #     return " ".join(text.strip().split())

# nlp.add_pipe(remove_whitespace_entities, after='ner')
   

    def main_train(self, model=None, output_dir=None, n_iter=100, TRAIN_DATA = []):
        """Load the model, set up the pipeline and train the entity recognizer."""
        if model is not None:
            nlp = spacy.load(model)  # load existing spaCy model
#             print("Loaded model '%s'" % model)
        else:
            nlp = spacy.blank('en')  # create blank Language class
#             print("Created blank 'en' model")
        nlp.add_pipe(self.remove_whitespace_entities, after='ner')
        # create the built-in pipeline components and add them to the pipeline
        # nlp.create_pipe works for built-ins that are registered with spaCy
        if 'ner' not in nlp.pipe_names:
            ner = nlp.create_pipe('ner')
            nlp.add_pipe(ner, last=True)
        # otherwise, get it so we can add labels
        else:
            ner = nlp.get_pipe('ner')
        # print(TRAIN_DATA)

        # add labels
        for _, annotations in TRAIN_DATA:
            for ent in annotations.get('entities'):
                ner.add_label(ent[2])

        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):  # only train NER
            optimizer = nlp.begin_training()
            for itn in range(n_iter):
                random.shuffle(TRAIN_DATA)
                losses = {}
                # batch up the examples using spaCy's minibatch
                batches = minibatch(TRAIN_DATA, size=compounding(4., 32., 1.001))
                for batch in batches:
                    texts, annotations = zip(*batch)
                    nlp.update(
                        texts,  # batch of texts
                        annotations,  # batch of annotations
                        drop=0.5,  # dropout - make it harder to memorise data
                        sgd=optimizer,  # callable to update weights
                        losses=losses)
#                 print('Losses', losses)

        # test the trained model
        for text, _ in TRAIN_DATA:
            doc = nlp(text)
#             print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
#             print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])

        # save model to output directory
        if output_dir is not None:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir()
            nlp.to_disk(output_dir)
#           print("Saved model to", output_dir)

            # test the saved model
#            print("Loading from", output_dir)
            nlp2 = spacy.load(output_dir)
            for text, _ in TRAIN_DATA:
                doc = nlp2(text)
#                print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
#                print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])
        return nlp
    # if __name__ == "__main__":
    #     main_train('en_core_web_lg')


