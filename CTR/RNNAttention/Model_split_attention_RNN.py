# -*- Encoding:UTF-8 -*-

import tensorflow as tf
import numpy as np
import argparse
from DataSet_split_attention_RNN import DataSet
import sys
import os
import heapq
import math
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq as seq2seq

def main():
    parser = argparse.ArgumentParser(description="Options")

    parser.add_argument('-dataName', action='store', dest='dataName', default='ml-100k')
    parser.add_argument('-negNum', action='store', dest='negNum', default=4, type=int)
    # parser.add_argument('-userLayer', action='store', dest='userLayer', default=[512, 64])
    parser.add_argument('-concat1Layer', action='store', dest='concat1Layer', default=[64])
    parser.add_argument('-concat2Layer', action='store', dest='concat2Layer', default=[64])
    parser.add_argument('-factor', action='store', dest='factor', default=64)
    parser.add_argument('-lr', action='store', dest='lr', default=0.001)
    parser.add_argument('-maxEpochs', action='store', dest='maxEpochs', default=50, type=int)
    parser.add_argument('-batchSize', action='store', dest='batchSize', default=256, type=int)
    parser.add_argument('-earlyStop', action='store', dest='earlyStop', default=5)
    parser.add_argument('-checkPoint', action='store', dest='checkPoint', default='./checkPoint/')
    parser.add_argument('-topK', action='store', dest='topK', default=10)

    args = parser.parse_args()

    classifier = Model(args)

    classifier.run()


class Model:
    def __init__(self, args):
        self.dataName = args.dataName
        self.dataSet = DataSet(self.dataName)
        self.shape = self.dataSet.shape
        self.maxRate = self.dataSet.maxRate

        self.train = self.dataSet.train
        self.test = self.dataSet.test

        self.factor = args.factor

        self.negNum = args.negNum
        self.testNeg = self.dataSet.getTestNeg(self.test, 99)
        self.add_embedding_matrix()

        self.add_placeholders()

        # self.userLayer = args.userLayer
        # self.itemLayer = args.itemLayer
        self.concat1Layer = args.concat1Layer
        self.concat2Layer = args.concat2Layer
        self.add_model()

        self.add_loss()

        self.lr = args.lr
        self.add_train_step()

        self.checkPoint = args.checkPoint
        self.init_sess()

        self.maxEpochs = args.maxEpochs
        self.batchSize = args.batchSize

        self.topK = args.topK
        self.earlyStop = args.earlyStop


    def add_placeholders(self):
        self.user = tf.placeholder(tf.int32 , [None ,])
        self.item = tf.placeholder(tf.int32 , [None ,])
        self.hist_item = tf.placeholder(tf.int32, shape=[None, None], name='hist_item')
        self.rnn_rnnHist = tf.placeholder(tf.int32, shape=[None, None], name='rnn_hist_item')
        self.sl = tf.placeholder(tf.int32, [None, ])
        self.rate = tf.placeholder(tf.float32)
        self.drop = tf.placeholder(tf.float32)

    def add_embedding_matrix(self):
        self.user_embedding =tf.get_variable(
                name='embeddingu',
                shape=[self.shape[0]+1, self.factor],
                initializer=tf.contrib.layers.xavier_initializer(),
                regularizer=tf.contrib.layers.l2_regularizer(0.001))

        self.item_embedding = tf.get_variable(
            name='embeddingi',
            shape=[self.shape[1], self.factor],
            initializer=tf.contrib.layers.xavier_initializer(),
            regularizer=tf.contrib.layers.l2_regularizer(0.001))
        self.item_embedding = tf.concat((tf.zeros(shape=[1 , self.factor]),
                                         self.item_embedding),0)

    def attention(self , queries , keys , keys_length):
        queries_hidden_units = queries.get_shape().as_list()[-1]
        queries = tf.tile(queries, [1, tf.shape(keys)[1]])
        queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])

        mul = tf.multiply(queries, keys)
        add = tf.reduce_sum(mul, axis=-1)
        outputs = tf.reshape(add, [-1, 1, tf.shape(keys)[1]])

        # Mask
        key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # [B, T]
        key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, T]
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

        # Scale
        outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

        # Activation
        outputs = tf.nn.softmax(outputs)  # [B, 1, T]

        # Weighted sum
        outputs = tf.matmul(outputs, keys)  # [B, 1, H]
        # sess.run(tf.global_variables_initializer())
        return outputs

    def add_model(self):
        user_input = tf.nn.embedding_lookup(self.user_embedding, self.user)
        item_input = tf.nn.embedding_lookup(self.item_embedding, self.item)

        itemHistss = tf.nn.embedding_lookup(self.item_embedding, self.hist_item)
        rnnItemHistss = tf.nn.embedding_lookup(self.item_embedding, self.rnn_rnnHist)
        hist = self.attention(user_input, itemHistss, self.sl)
        # hist = tf.nn.dropout(hist , self.drop)

        hist = tf.reshape(hist, [-1, self.factor])

        state_size = 64
        with tf.name_scope('model'):
            # self.cell = rnn.BasicRNNCell(state_size)
            self.cell = rnn.BasicLSTMCell(state_size)
            # self.cell = rnn.DropoutWrapper(self.cell , self.drop)


            self.cell = rnn.MultiRNNCell([self.cell] * 4)
            self.initial_state = self.cell.zero_state(tf.shape(self.hist_item)[0], tf.float32)

            # embedding = tf.get_variable('embedding', [self.dataSet.shape[1], state_size])
            # inputs = tf.nn.embedding_lookup(embedding, self.hist_item)
            outputs, last_state = tf.nn.dynamic_rnn(self.cell, rnnItemHistss, initial_state=self.initial_state)

        preNextItem = tf.concat([user_input , hist] , axis=1)
        with tf.name_scope("PreNItem"):
            for idx in self.concat1Layer:
                preNextItem = tf.layers.dense(preNextItem, idx, activation=tf.nn.relu)
                # preNextItem = tf.nn.dropout(preNextItem , self.drop)

        with tf.name_scope("concat_rnnAttention"):
            # last_state = tf.nn.dropout(last_state , self.drop)
            print(tf.shape(last_state))
            preNextItem = tf.concat([preNextItem, last_state[0][0]+last_state[0][1]+last_state[0][2]+last_state[0][3]] , 1)
            preNextItem = tf.layers.dense(preNextItem, self.factor, activation=tf.nn.relu)
            # preNextItem = tf.add(preNextItem , last_state[0])
            # preNextItem = tf.nn.dropout(preNextItem, self.drop)

        con = tf.concat([preNextItem , item_input] , axis=1)
        with tf.name_scope("PreCos"):
            for idx in self.concat2Layer:
                con = tf.layers.dense(con, idx, activation=tf.nn.relu)
                # con = tf.nn.dropout(con, self.drop)

        self.y_ = tf.layers.dense(con, 1, activation=None)
        self.y_ = tf.squeeze(self.y_ , 1)

        self.logPred = tf.sigmoid(self.y_)

    def add_loss(self):
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_, labels=self.rate))

        # regLoss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        # self.loss = self.loss + 0.000001 * regLoss

    def add_train_step(self):
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_step = optimizer.minimize(self.loss)

    def init_sess(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.global_variables_initializer())

    def run(self):
        best_hr = -1
        best_NDCG = -1
        best_epoch = -1
        print("Start Training!")
        for epoch in range(self.maxEpochs):
            print("="*20+"Epoch ", epoch, "="*20)
            self.run_epoch(self.sess)
            print('='*50)
            print("Start Evaluation!")
            hr, NDCG = self.evaluate(self.sess, self.topK)
            print("Epoch ", epoch, "HR: {}, NDCG: {}".format(hr, NDCG))
            if hr > best_hr:
                best_hr = hr
                best_NDCG = NDCG
                best_epoch = epoch
            if epoch - best_epoch > self.earlyStop:
                print("Normal Early stop!")
                break
            print("="*20+"Epoch ", epoch, "End"+"="*20)
        print("Best hr: {}, NDCG: {}, At Epoch {}".format(best_hr, best_NDCG, best_epoch))
        print("Training complete!")

    def nextDataInput(self, users , itemPoss=None , isTest=True):
        a_item_num = 15
        rnn_item_Num = 15
        u , i , a_sl , rnn_sl = [],[],[],[]
        if isTest:
            for user in users:
                a_sl.append(int(min(a_item_num , len(self.dataSet.userHistDict[user]))))
                rnn_sl.append(int(min(rnn_item_Num, len(self.dataSet.userHistDict[user]))))
            max_asl = int(max(a_sl))
            max_rnnsl = int(max(rnn_sl))
            hist = np.zeros([len(users), max_asl], np.int32)
            rnnHist = np.zeros([len(users), max_rnnsl], np.int32)
            for idx in range(len(users)):
                user = users[idx]
                userHists = self.dataSet.userHistDict[user]
                for index in range(a_sl[idx]):
                    hist[idx][index] = userHists[-(a_sl[idx] - index)]
                for index in range(rnn_sl[idx]):
                    rnnHist[idx][max_rnnsl-rnn_sl[idx]+index] = userHists[-(rnn_sl[idx] - index)]
                
        else:
            for user , itemPos in zip(users , itemPoss):
                a_sl.append(int(min(a_item_num , itemPos)))
                rnn_sl.append(int(min(rnn_item_Num, itemPos)))
            max_asl = int(max(a_sl))
            max_rnnsl = int(max(rnn_sl))
            hist = np.zeros([len(users), max_asl], np.int32)
            rnnHist = np.zeros([len(users), max_rnnsl], np.int32)
            for idx in range(len(users)):
                user = users[idx]
                userHists = self.dataSet.userHistDict[user]
                a_start = itemPoss[idx] - int(min(a_item_num , a_sl[idx]))
                rnn_start = itemPoss[idx] - int(min(rnn_item_Num , rnn_sl[idx]))
                end = itemPoss[idx]
                for index in range(a_start , end):
                    hist[idx][index - a_start] = userHists[index]
                for index in range(rnn_start, end):
                    rnnHist[idx][max_rnnsl - end + index] = userHists[index]

        return hist , a_sl , rnnHist

    def run_epoch(self, sess, verbose=10):
        train_u, train_i , train_r , train_i_pos= self.dataSet.getInstances(self.train, self.negNum)
        train_len = len(train_u)
        shuffled_idx = np.random.permutation(np.arange(train_len))
        train_u = train_u[shuffled_idx]
        train_i = train_i[shuffled_idx]
        train_r = train_r[shuffled_idx]
        train_i_pos = train_i_pos[shuffled_idx]

        num_batches = len(train_u) // self.batchSize + 1

        losses = []
        for i in range(num_batches):
            min_idx = i * self.batchSize
            max_idx = np.min([train_len, (i+1)*self.batchSize])
            train_u_batch = train_u[min_idx: max_idx]
            train_i_batch = train_i[min_idx: max_idx]
            train_r_batch = train_r[min_idx: max_idx]
            train_pos_batch = train_i_pos[min_idx: max_idx]

            userHists, histLen , rnnHist = self.nextDataInput(train_u_batch, train_pos_batch, isTest=False)

            feed_dict = self.create_feed_dict(train_u_batch, train_i_batch ,userHists , histLen, rnnHist, train_r_batch , drop=0.8)
            _, tmp_loss = sess.run([self.train_step, self.loss], feed_dict=feed_dict)

            # q , k , o = sess.run([self.que , self.key , self.out] , feed_dict=feed_dict)
            # print("参数：")
            # print(q)
            # print(k)
            # print(o)

            losses.append(tmp_loss)
            if verbose and i % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                    i, num_batches, np.mean(losses[-verbose:])
                ))
                sys.stdout.flush()

        loss = np.mean(losses)
        print("\nMean loss in this epoch is: {}".format(loss))
        return loss

    def create_feed_dict(self, u, i,histItem,histLen,rnnHist, r=None, drop=None):
        return {self.user: u,
                self.item: i,
                self.hist_item:histItem,
                self.sl:histLen,
                self.rnn_rnnHist:rnnHist,
                self.rate: r,
                self.drop: drop}

    def create_feed_dict1(self, u, i,histItem,histLen,rnnHist, drop=None):
        return {self.user: u,
                self.item: i,
                self.hist_item: histItem,
                self.rnn_rnnHist: rnnHist,
                self.sl: histLen,
                self.drop: drop}

    def evaluate(self, sess, topK):
        def getHitRatio(ranklist, targetItem):
            for item in ranklist:
                if item == targetItem:
                    return 1
            return 0
        def getNDCG(ranklist, targetItem):
            for i in range(len(ranklist)):
                item = ranklist[i]
                if item == targetItem:
                    return math.log(2) / math.log(i+2)
            return 0


        hr =[]
        NDCG = []
        testUser = self.testNeg[0]
        testItem = self.testNeg[1]
        for i in range(len(testUser)):
            target = testItem[i][0]
            userHists, histLen , rnnHist = self.nextDataInput(testUser[i])

            feed_dict = self.create_feed_dict1(testUser[i], testItem[i] , userHists , histLen,rnnHist , drop=1.0)
            predict = sess.run(self.logPred, feed_dict=feed_dict)

            item_score_dict = {}

            for j in range(len(testItem[i])):
                item = testItem[i][j]
                item_score_dict[item] = predict[j]

            ranklist = heapq.nlargest(topK, item_score_dict, key=item_score_dict.get)

            tmp_hr = getHitRatio(ranklist, target)
            tmp_NDCG = getNDCG(ranklist, target)
            hr.append(tmp_hr)
            NDCG.append(tmp_NDCG)
        return np.mean(hr), np.mean(NDCG)

if __name__ == '__main__':
    main()
