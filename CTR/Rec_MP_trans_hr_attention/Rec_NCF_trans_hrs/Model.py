# -*- Encoding:UTF-8 -*-

import tensorflow as tf
import numpy as np
import argparse
from Rec_NCF_trans_hrs.DataSet import DataSet
import sys
import os
import heapq
import math


def main():
    parser = argparse.ArgumentParser(description="Options")

    parser.add_argument('-dataName', action='store', dest='dataName', default='ml-1m')
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
        self.sl = tf.placeholder(tf.int32, [None, ])
        self.rate = tf.placeholder(tf.float32)
        self.drop = tf.placeholder(tf.float32)

    def add_embedding_matrix(self):
        self.user_embedding =tf.get_variable(
                name='embeddingu',
                shape=[self.shape[0], self.factor],
                initializer=tf.contrib.layers.xavier_initializer(),
                regularizer=tf.contrib.layers.l2_regularizer(0.001))

        self.item_embedding = tf.get_variable(
            name='embeddingi',
            shape=[self.shape[1], self.factor],
            initializer=tf.contrib.layers.xavier_initializer(),
            regularizer=tf.contrib.layers.l2_regularizer(0.001))

    def attention(self , queries , keys , keys_length):
        queries_hidden_units = queries.get_shape().as_list()[-1]
        queries = tf.tile(queries, [1, tf.shape(keys)[1]])
        queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])
        din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
        # print(sess.run(din_all))

        d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att')
        d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att')
        d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att')
        d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])
        outputs = d_layer_3_all

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
        hist = self.attention(user_input, itemHistss, self.sl)

        hist = tf.layers.batch_normalization(inputs=hist)
        hist = tf.reshape(hist, [-1, self.factor])


        preNextItem = tf.concat([user_input , hist] , axis=1)
        with tf.name_scope("PreNItem"):
            for idx in self.concat1Layer:
                preNextItem = tf.layers.dense(preNextItem, idx, activation=tf.nn.relu)

        con = tf.concat([preNextItem , item_input] , axis=1)
        with tf.name_scope("PreCos"):
            for idx in self.concat2Layer:
                con = tf.layers.dense(con, idx, activation=tf.nn.relu)

        self.y_ = tf.layers.dense(con, 1, activation=None)
        self.y_ = tf.squeeze(self.y_ , 1)

        # self.y_ = tf.maximum(1e-6, self.y_)
        self.logPred = tf.sigmoid(self.y_)

    def add_loss(self):
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_, labels=self.rate))

    def add_train_step(self):
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_step = optimizer.minimize(self.loss)

    def init_sess(self):
        self.sess = tf.Session()
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
            if hr > best_hr or NDCG > best_NDCG:
                best_hr = hr
                best_NDCG = NDCG
                best_epoch = epoch
            if epoch - best_epoch > self.earlyStop:
                print("Normal Early stop!")
                break
            print("="*20+"Epoch ", epoch, "End"+"="*20)
        print("Best hr: {}, NDCG: {}, At Epoch {}".format(best_hr, best_NDCG, best_epoch))
        print("Training complete!")

    #训练集不需要位置，所有itemPoss初始为None,可不传值
    def nextDataInput(self, users , itemPoss=None , isTest=True):
        #复杂度太高，每个窗口只取10个物品
        u , i , sl = [],[],[]
        if isTest:
            for user in users:
                sl.append(int(min(10 , len(self.dataSet.userHistDict[user]))))   #所有历史记录
            max_sl = int(max(sl))
            hist = np.zeros([len(users), max_sl], np.int32)
            for idx in range(len(users)):
                user = users[idx]
                userHists = self.dataSet.userHistDict[user]
                for index in range(sl[idx]):   #取最近的10个物品来预测下一个物品
                    hist[idx][index] = userHists[-(sl[idx] - index)]
                
        else:
            for user , itemPos in zip(users , itemPoss):   #itemPos表示当前物品前面共有多少个物品历史记录
                sl.append(int(min(10 , itemPos)))
            max_sl = int(max(sl))
            hist = np.zeros([len(users), max_sl], np.int32)
            for idx in range(len(users)):
                user = users[idx]
                userHists = self.dataSet.userHistDict[user]
                start = itemPoss[idx] - 10
                end = itemPoss[idx]
                for index in range(start , end):     #训练集只取当前物品之前的历史物品集合
                    hist[idx][index - start] = userHists[index]    #hist从0开始取，userHists不是从0开始取

        return hist , sl

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

            userHists, histLen = self.nextDataInput(train_u_batch, train_pos_batch, isTest=False)  #获取历史

            feed_dict = self.create_feed_dict(train_u_batch, train_i_batch ,userHists , histLen, train_r_batch)
            _, tmp_loss = sess.run([self.train_step, self.loss], feed_dict=feed_dict)

            losses.append(tmp_loss)
            if verbose and i % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                    i, num_batches, np.mean(losses[-verbose:])
                ))
                sys.stdout.flush()

        loss = np.mean(losses)
        print("\nMean loss in this epoch is: {}".format(loss))
        return loss

    def create_feed_dict(self, u, i,histItem,histLen, r=None, drop=None):
        return {self.user: u,
                self.item: i,
                self.hist_item:histItem,
                self.sl:histLen,
                self.rate: r,
                self.drop: drop}

    def create_feed_dict1(self, u, i,histItem,histLen, drop=None):
        return {self.user: u,
                self.item: i,
                self.hist_item: histItem,
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
            userHists, histLen = self.nextDataInput(testUser[i])

            feed_dict = self.create_feed_dict1(testUser[i], testItem[i] , userHists , histLen , drop=1.0)
            predict = sess.run(self.y_, feed_dict=feed_dict)

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
