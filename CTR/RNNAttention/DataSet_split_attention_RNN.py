# -*- Encoding:UTF-8 -*-

import numpy as np
import sys

class DataSet(object):
    def __init__(self, fileName):
        self.data, self.shape = self.getData(fileName)
        self.train, self.test = self.getTrainTest()
        self.trainDict = self.getTrainDict()

    def getData(self, fileName):
        if fileName == 'ml-1m':
            print("Loading ml-1m data set...")
            data = []
            filePath = './Data/ml-1m/ratings.dat'
            u = 0
            i = 0
            maxr = 0.0
            with open(filePath, 'r') as f:
                for line in f:
                    if line:
                        lines = line[:-1].split("::")
                        user = int(lines[0])
                        movie = int(lines[1])
                        score = float(lines[2])
                        time = int(lines[3])
                        data.append((user, movie, score, time))
                        if user > u:
                            u = user
                        if movie > i:
                            i = movie
                        if score > maxr:
                            maxr = score
            self.maxRate = maxr
            print("Loading Success!\n"
                  "Data Info:\n"
                  "\tUser Num: {}\n"
                  "\tItem Num: {}\n"
                  "\tData Size: {}".format(u, i, len(data)))
            return data, [u, i]
        elif fileName == 'ml-100k':
            print("Loading ml-100k data set...")
            data = []
            filePath = './Data/ml-100k/u.data'
            u = 0
            i = 0
            maxr = 0.0
            with open(filePath, 'r') as f:
                for line in f:
                    if line:
                        lines = line[:-1].split("\t")
                        user = int(lines[0])
                        movie = int(lines[1])
                        score = float(lines[2])
                        time = int(lines[3])
                        data.append((user, movie, score, time))
                        if user > u:
                            u = user
                        if movie > i:
                            i = movie
                        if score > maxr:
                            maxr = score
            self.maxRate = maxr
            print("Loading Success!\n"
                  "Data Info:\n"
                  "\tUser Num: {}\n"
                  "\tItem Num: {}\n"
                  "\tData Size: {}".format(u, i, len(data)))
            return data, [u, i]
        elif fileName == 'hak':
            print("Loading Ahome_and_kitchen data set...")
            data = []
            filePath = './Data/Ahome_and_kitchen/AMazon_home_and_kitchen.dat'
            u = 0
            i = 0
            maxr = 0.0
            with open(filePath, 'r') as f:
                for line in f:
                    if line:
                        lines = line[:-1].split(",")
                        user = int(lines[0])
                        movie = int(lines[1])
                        score = float(lines[2])
                        time = int(lines[3])
                        data.append((user, movie, score, time))
                        if user > u:
                            u = user
                        if movie > i:
                            i = movie
                        if score > maxr:
                            maxr = score
            self.maxRate = maxr
            print("Loading Success!\n"
                  "Data Info:\n"
                  "\tUser Num: {}\n"
                  "\tItem Num: {}\n"
                  "\tData Size: {}".format(u, i, len(data)))
            return data, [u, i]
        else:
            print("Current data set is not support!")
            sys.exit()

    def getTrainTest(self):
        data = self.data
        data = sorted(data, key=lambda x: (x[0], x[3]))
        self.userHistDict = {}
        train = []
        test = []
        for i in range(len(data)-1):
            user = data[i][0]
            item = data[i][1]
            rate = data[i][2]
            if data[i][0] != data[i+1][0]:
                test.append((user, item, rate))
            else:
                if user not in self.userHistDict.keys():
                    self.userHistDict[user] = []
                self.userHistDict[user].append(item)

                train.append((user, item, rate , self.userHistDict[user].index(item)))

        test.append((data[-1][0], data[-1][1], data[-1][2]))
        return train, test

    def getTrainDict(self):
        dataDict = {}
        for i in self.train:
            dataDict[(i[0], i[1])] = i[2]
        return dataDict

    def getInstances(self, data, negNum):
        user = []
        item = []
        rate = []
        itemPos = []
        for i in data:
            user.append(i[0])
            item.append(i[1])
            rate.append(1.0)
            itemPos.append(i[3])
            for t in range(negNum):
                j = np.random.randint(self.shape[1])
                while j==i[1]:
                    j = np.random.randint(self.shape[1])
                user.append(i[0])
                item.append(j)
                rate.append(0.0)
                itemPos.append(i[3])
        return np.array(user), np.array(item), np.array(rate) , np.array(itemPos)

    def getTestNeg(self, testData, negNum):
        user = []
        item = []
        for s in testData:
            tmp_user = []
            tmp_item = []
            u = s[0]
            i = s[1]
            tmp_user.append(u)
            tmp_item.append(i)
            for t in range(negNum):
                j = np.random.randint(self.shape[1])
                while (u, j) in self.trainDict:
                    j = np.random.randint(self.shape[1])
                tmp_user.append(u)
                tmp_item.append(j)
            user.append(tmp_user)
            item.append(tmp_item)
        return [np.array(user), np.array(item)]
