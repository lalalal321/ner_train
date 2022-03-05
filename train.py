# encoding: utf-8
"""
@author: zhou
@time: 2022/2/20 16:49
@file: test.py.py
@desc: 
"""

# Load build-in corpus.
import pickle

from kashgari.embeddings import BertEmbedding
from kashgari.tasks.labeling import BiLSTM_CRF_Model

x = pickle.load(open('train_x.pickle', 'rb'))
y = pickle.load(open('train_y.pickle', 'rb'))

train_x, train_y = x[0:7000], y[0:7000]
valid_x, valid_y = train_x, train_y
test_x, test_y = x[7000:7300], y[7000:7300]


def bert():
    bert_embed = BertEmbedding("uncased_L-12_H-768_A-12")
    model = BiLSTM_CRF_Model(bert_embed, sequence_length=128)
    model.fit(train_x, train_y, valid_x, valid_y, epochs=5)
    model.evaluate(test_x, test_y)


def no_bert():
    model = BiLSTM_CRF_Model()
    model.fit(train_x, train_y, valid_x, valid_y, epochs=5)
    # Evaluate the model
    model.evaluate(test_x, test_y)



if __name__ == '__main__':
    bert()




# Model data will save to `saved_ner_model` folder
# model.save('saved_ner_model')

# Load saved model
# loaded_model = BiLSTM_Model.load_model('saved_ner_model')
# loaded_model.predict(test_x[:10])

# To continue training, compile the newly loaded model first
# loaded_model.compile_model()
# model.fit(train_x, train_y, valid_x, valid_y)
