from flask import *
from flask_restful import *
from flask_restful import reqparse
import wandb
import nltk
from nltk.stem.porter import *
from torch.nn import *
from torch.optim import *
import numpy as np
import pandas as pd
import torch, torchvision
import random
from tqdm import *
from torch.utils.data import Dataset, DataLoader

stemmer = PorterStemmer()
PROJECT_NAME = "kickstarter-NLP-V4"
device = "cuda"
words = torch.load("./words.pt")
labels = torch.load("./labels.pt")


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_words, words):
    tokenized_words = [stem(w) for w in tokenized_words]
    bag = np.zeros(len(words))
    for idx, w in enumerate(words):
        if w in tokenized_words:
            bag[idx] = 1.0
    return bag


class Model(Module):
    def __init__(self):
        super().__init__()
        self.activation = ReLU()
        self.dropout = Dropout()
        self.hidden = 512
        self.batchnorm = BatchNorm1d(self.hidden)
        self.linear1 = Linear(len(words), self.hidden)
        self.linear2 = Linear(self.hidden, self.hidden)
        self.linear3 = Linear(self.hidden, self.hidden)
        self.linear4 = Linear(self.hidden, self.hidden)
        self.linear5 = Linear(self.hidden, self.hidden)
        self.output = Linear(self.hidden, len(labels))

    def forward(self, X):
        preds = self.linear1(X)
        preds = self.activation(self.linear2(preds))
        preds = self.batchnorm(self.dropout(self.activation(self.linear3(preds))))
        preds = self.activation(self.linear4(preds))
        preds = self.activation(self.linear5(preds))
        preds = self.output(preds)
        return preds


model = Model()
model.load_state_dict(torch.load("./model-sd.pt"))
app = Flask(__name__)
app.debug = True
app.secret_key = "secret_key"
api = Api(app)

args = reqparse.RequestParser()
args.add_argument("chat", type=str, help="chat is required", required=True)


class NLP_API(Resource):
    def get(self):
        data = args.parse_args()
        data = data["chat"]
        data = tokenize(data)
        new_data = []
        for d in data:
            new_data.append(stem(d))
        data = new_data
        data = bag_of_words(data, words)
        model.eval()
        pred = model(data)
        print(pred)
        pred = torch.argmax(pred[0])
        pred = int(pred)
        if pred == 0:
            return {"Response": False}
        return {"Response": True}


api.add_response(NLP_API, "/")
if __name__ == "__main__":
    app.run(debug=True)
