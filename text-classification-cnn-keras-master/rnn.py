# -*- coding: utf-8 -*-
from keras.layers import Input, Dense, Embedding, Conv1D, MaxPool1D, GRU, Bidirectional ,LSTM
from keras.layers import Flatten, Dropout
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model

import os
from data.cnews_loader import read_vocab, read_category, process_file, build_vocab

base_dir = 'data/cnews'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

words, word_to_id = read_vocab(vocab_dir)

seq_length = 600
categories, cat_to_id = read_category()
vocab_size = len(words)
embedding_dim = 64      # 词向量维度
num_classes = 10        # 类别数
#vocab_size = 5000       # 词汇表达小

num_layers= 2           # 隐藏层层数
hidden_dim = 128        # 隐藏层神经元
rnn = 'gru'             # lstm 或 gru

drop = 0.2
learning_rate = 1e-3    # 学习率

batch_size = 128         # 每批训练大小
epochs = 10          # 总迭代轮次


if not os.path.exists(vocab_dir):#建立词汇表
    build_vocab(train_dir, vocab_dir, vocab_size)

x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, seq_length)
x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, seq_length)
x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, seq_length)

inputs = Input(shape=(seq_length,), dtype='int64')

embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=seq_length)(inputs)
gru = Bidirectional(GRU(128))(embedding)

#pre_dense = Flatten()(gru)
dense = Dense(128, activation='relu')(gru)
dense = Dropout(drop)(dense)
output = Dense(units=10,activation='softmax')(dense)

model = Model(inputs=inputs, outputs=output)

checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
print("Traning Model...")
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_data=(x_val, y_val))