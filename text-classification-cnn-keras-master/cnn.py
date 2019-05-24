# -*- coding: utf-8 -*-
from keras.layers import Input, Dense, Embedding, Conv1D, MaxPool1D
from keras.layers import Flatten, Dropout
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from keras.preprocessing.text import Tokenizer
import numpy as np
import sklearn
from sklearn import metrics
from attention_layer import *

import os
from data.cnews_loader import read_vocab, read_category, process_file, build_vocab

base_dir = 'data/cnews'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'vocab.txt')

words, word_to_id = read_vocab(vocab_dir)

seq_length = 600
categories, cat_to_id = read_category()
vocab_size = len(words)
embedding_dim = 300
filter_sizes = [5]
num_filters = 256
drop = 0.5
epochs = 20
batch_size = 64

'''if not os.path.exists(vocab_dir):#建立词汇表
    build_vocab(train_dir, vocab_dir, vocab_size)'''

x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, seq_length)
x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, seq_length)
x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, seq_length)

#w2v
'''
GLOVE_DIR = 'embedding'
#vocab={} # 词汇表为数据预处理后得到的词汇字典

# 构建词向量索引字典
## 读入词向量文件，文件中的每一行的第一个变量是单词，后面的一串数字对应这个词的词向量

f = open(os.path.join(GLOVE_DIR, 'sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'))
## 获取词向量的维度,l表示单词数，w为某个单词转化为词向量后的维度,注意，部分预训练好的词向量的第一行并不是该词向量的维度
#l,w=f.readline().split()
## 创建词向量索引字典
embeddings_index={}
for line in f:
    ## 读取词向量文件中的每一行
    values=line.split()
    ## 获取当前行的词
    word=values[0]
    ## 获取当前词的词向量
    coefs=np.asarray(values[1:],dtype="float32")
    ## 将读入的这行词向量加入词向量索引字典
    embeddings_index[word]=coefs
f.close()

# 构建词向量矩阵，预训练的词向量中没有出现的词用0向量表示
## 创建一个0矩阵，这个向量矩阵的大小为（词汇表的长度+1，词向量维度）
embedding_matrix=np.zeros((vocab_size+1,embedding_dim))
## 遍历词汇表中的每一项
for i,word in enumerate(words):
    ## 在词向量索引字典中查询单词word的词向量
    embedding_vector=embeddings_index.get(word)
    ## 判断查询结果，如果查询结果不为空,用该向量替换0向量矩阵中下标为i的那个向量
    if embedding_vector is not None:
        embedding_matrix[i]=embedding_vector

'''



inputs = Input(shape=(seq_length,), dtype='int64')
#embedding=Embedding(vocab_size+1, embedding_dim, input_length = seq_length, weights = [embedding_matrix], trainable = False)(inputs)

embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=seq_length)(inputs)
#embedding = Position_Embedding()(embedding)
conv = Conv1D(num_filters, filter_sizes, activation='relu')(embedding)
gru = Attention(8,16)([conv, conv, conv])
maxpool = MaxPool1D(5)(gru)

pre_dense = Flatten()(maxpool)
dense = Dense(128, activation='relu')(pre_dense)
dense = Dropout(drop)(dense)
output = Dense(units=10,activation='softmax')(dense)

model = Model(inputs=inputs, outputs=output)

checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
print("Traning Model...")
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_data=(x_val, y_val))