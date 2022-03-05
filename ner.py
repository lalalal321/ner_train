# encoding: utf-8
"""
@author: zhou
@time: 2022/3/5 13:48
@file: 3.ner.py
@desc: 
"""

# Setup macros
SEQUENCE_LENGTH = 200
EPOCHS = 30
EARL_STOPPING_PATIENCE = 5
REDUCE_RL_PATIENCE = 5

BATCH_SIZE = 64

# EMBEDDING_FOLDER = './embeddings'
TF_LOG_FOLDER = 'log/tf_dir'
LOG_FILE_PATH = 'log/ner_training_log.json'

import json
import os
import pickle

from kashgari.callbacks import EvalCallBack
from kashgari.embeddings import BertEmbedding
# from kashgari.tasks.labeling import BiGRU_Model, BiGRU_CRF_Model
from kashgari.tasks.labeling import BiLSTM_Model, BiLSTM_CRF_Model
from tensorflow import keras

train_x = pickle.load(open('train_x.pickle', 'rb'))
train_y = pickle.load(open('train_y.pickle', 'rb'))
valid_x, valid_y = train_x, train_y

test_x = pickle.load(open('test_x.pickle', 'rb'))
test_y = pickle.load(open('test_x.pickle', 'rb'))

# Google Bert
bert_embed = BertEmbedding("uncased_L-12_H-768_A-12")

embeddings = [
    ('Bert', bert_embed),
    ('Bare', None)
]

model_classes = [
    ('BiLSTM', BiLSTM_Model),
    ('BiLSTM_CRF', BiLSTM_CRF_Model),
    # ('BiGRU', BiGRU_Model),
    # ('BiGRU_CRF', BiGRU_CRF_Model)
]

for embed_name, embed in embeddings:
    for model_name, MOEDL_CLASS in model_classes:
        run_name = f"{embed_name}-{model_name}"
        model = MOEDL_CLASS(embed, sequence_length=SEQUENCE_LENGTH)

        early_stop = keras.callbacks.EarlyStopping(patience=EARL_STOPPING_PATIENCE)
        reduse_lr_callback = keras.callbacks.ReduceLROnPlateau(factor=0.1,
                                                               patience=REDUCE_RL_PATIENCE)

        eval_callback = EvalCallBack(kash_model=model,
                                     x_data=valid_x,
                                     y_data=valid_y,
                                     step=1)

        tf_board = keras.callbacks.TensorBoard(
            log_dir=os.path.join(TF_LOG_FOLDER, run_name),
            update_freq=1000
        )

        callbacks = [early_stop, reduse_lr_callback, eval_callback, tf_board]

        model.fit(train_x,
                  train_y,
                  valid_x,
                  valid_y,
                  callbacks=callbacks,
                  epochs=EPOCHS)

        if os.path.exists(LOG_FILE_PATH):
            logs = json.load(open(LOG_FILE_PATH, 'r'))
        else:
            logs = {}

        logs[run_name] = eval_callback.logs

        with open(LOG_FILE_PATH, 'w') as f:
            f.write(json.dumps(logs, indent=2))
