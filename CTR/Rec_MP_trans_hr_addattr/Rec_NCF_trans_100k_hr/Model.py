# -*- Encoding:UTF-8 -*-

import tensorflow as tf
import numpy as np
import argparse
from Rec_NCF_trans_100k_hr.DataSet import DataSet
import sys
import os
import heapq
import math


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

        self.userFeatAttr = tf.convert_to_tensor(self.dataSet.userFeatAttr , tf.float32)
        self.movieFeatAttr = tf.convert_to_tensor(self.dataSet.movieFeatAttr , tf.float32)

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
        self.next_item = tf.placeholder(tf.int32 , [None ,])
        self.rate = tf.placeholder(tf.float32)
        self.drop = tf.placeholder(tf.float32)

    def add_embedding_matrix(self):
        # self.user_item_embedding = tf.convert_to_tensor(self.dataSet.getEmbedding())
        # self.item_user_embedding = tf.transpose(self.user_item_embedding)
        self.user_embedding =tf.get_variable(
                name='embeddingu',
                shape=[self.shape[0], self.factor],
                initializer=tf.contrib.layers.xavier_initializer(),
                regularizer=tf.contrib.layers.l2_regularizer(0.0))
        self.item_embedding = tf.get_variable(
            name='embeddingi',
            shape=[self.shape[1], self.factor],
            initializer=tf.contrib.layers.xavier_initializer(),
            regularizer=tf.contrib.layers.l2_regularizer(0.0))

    def add_model(self):
        user_input = tf.nn.embedding_lookup(self.user_embedding, self.user)
        item_input = tf.nn.embedding_lookup(self.item_embedding, self.item)
        next_item_input = tf.nn.embedding_lookup(self.item_embedding, self.next_item)
        user_input_attr = tf.nn.embedding_lookup(self.userFeatAttr,self.user)
        item_input_attr = tf.nn.embedding_lookup(self.movieFeatAttr,self.item)
        next_item_input_attr = tf.nn.embedding_lookup(self.movieFeatAttr,self.next_item)

        userConcat = tf.concat([user_input , user_input_attr] , axis=1)
        user_input_new = tf.layers.dense(userConcat, 64, activation=None)

        itemConcat = tf.concat([item_input , item_input_attr] , axis=1)
        item_input_new = tf.layers.dense(itemConcat, 64, activation=None)

        nextitemConcat = tf.concat([next_item_input, next_item_input_attr], axis=1)
        next_item_input_new = tf.layers.dense(nextitemConcat, 64, activation=None)

        concatOneLayers = tf.concat([user_input_new , item_input_new] , axis=1)
        with tf.name_scope("concat1"):
            for idx in self.concat1Layer:
                concatOneLayers = tf.layers.dense(concatOneLayers, idx, activation=tf.nn.relu)

        concatTwoLayers = tf.concat([concatOneLayers , next_item_input_new] , axis=1)
        with tf.name_scope("concat2"):
            for idx in self.concat2Layer:
                concatTwoLayers = tf.layers.dense(concatTwoLayers, idx, activation=tf.nn.relu)

        self.y_ = tf.layers.dense(concatTwoLayers, 1, activation=None)


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

    def run_epoch(self, sess, verbose=10):
        train_u, train_i, train_next_i , train_r = self.dataSet.getInstances(self.train, self.negNum)
        train_len = len(train_u)
        shuffled_idx = np.random.permutation(np.arange(train_len))
        train_u = train_u[shuffled_idx]
        train_i = train_i[shuffled_idx]
        train_next_i = train_next_i[shuffled_idx]
        train_r = train_r[shuffled_idx]

        num_batches = len(train_u) // self.batchSize + 1

        losses = []
        for i in range(num_batches):
            min_idx = i * self.batchSize
            max_idx = np.min([train_len, (i+1)*self.batchSize])
            train_u_batch = train_u[min_idx: max_idx]
            train_i_batch = train_i[min_idx: max_idx]
            train_next_i_batch = train_next_i[min_idx: max_idx]
            train_r_batch = train_r[min_idx: max_idx]

            feed_dict = self.create_feed_dict(train_u_batch, train_i_batch , train_next_i_batch, train_r_batch)
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

    def create_feed_dict(self, u, i ,next_i, r=None, drop=None):
        return {self.user: u,
                self.item: i,
                self.next_item:next_i,
                self.rate: r,
                self.drop: drop}

    def create_feed_dict1(self, u, i,next_i, drop=None):
        return {self.user: u,
                self.item: i,
                self.next_item: next_i,
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
        testnextItem = self.testNeg[2]
        for i in range(len(testUser)):
            target = testnextItem[i][0]

            feed_dict = self.create_feed_dict1(testUser[i], testItem[i] , testnextItem[i])
            predict = sess.run(self.logPred, feed_dict=feed_dict)

            item_score_dict = {}

            for j in range(len(testnextItem[i])):
                item = testnextItem[i][j]
                item_score_dict[item] = predict[j]

            ranklist = heapq.nlargest(topK, item_score_dict, key=item_score_dict.get)

            tmp_hr = getHitRatio(ranklist, target)
            tmp_NDCG = getNDCG(ranklist, target)
            hr.append(tmp_hr)
            NDCG.append(tmp_NDCG)
        return np.mean(hr), np.mean(NDCG)

if __name__ == '__main__':
    main()
