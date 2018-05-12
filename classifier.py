from urlparse import urlparse
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_extraction.text import CountVectorizer

def tokenize(url):
    parsed_url = urlparse(url)
    tokens = parsed_url.netloc.split('.')
    tokens.append(parsed_url.path[1:][:-1].split('/'))

    return tokens

class Tokenizer(object):
     # def __init__(self):
    def __call__(self, doc):
        parsed_url = urlparse(doc)
        tokens = parsed_url.netloc.split('.')
        tokens.append(parsed_url.path[1:][:-1].split('/'))
        return tokens


def initialize_classifier():
	clf = Pipeline([
        ('union', FeatureUnion(
        	transformer_list=[
        		('vectorizer', CountVectorizer(ngram_range=(1, 4), tokenizer=Tokenizer(), analyzer='char_wb')),
            ],
        )),

        ('classifier', LR())
    ])
	return clf
