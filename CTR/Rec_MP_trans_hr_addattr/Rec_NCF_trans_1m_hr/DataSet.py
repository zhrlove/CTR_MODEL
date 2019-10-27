# -*- Encoding:UTF-8 -*-

import numpy as np
import sys
from Rec_NCF_trans_1m_hr.UserFeatures import *
from Rec_NCF_trans_1m_hr.MovieFeatures import *


class DataSet(object):
    def __init__(self, fileName):
        self.data, self.shape = self.getData(fileName)
        self.getAttr()
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
        elif fileName == 'mt':
            print("Loading mt data set...")
            data = []
            filePath = './Data/mt/movieTweews_ratings.dat'
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
        elif fileName == 'AMusic':
            print("Loading AMusic data set...")
            data = []
            filePath = './Data/AMusic/AMusic_ratings.dat'
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
        elif fileName == 'Aapp':
            print("Loading Aapps_for_android data set...")
            data = []
            filePath = './Data/Aapps_for_android/AMazon_apps_for_android.dat'
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
        elif fileName == 'Acds':
            print("Loading Acds_and_vinyl data set...")
            data = []
            filePath = './Data/Acds_and_vinyl/AMazon_cds_and_vinyl.dat'
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
        elif fileName == 'admu':
            print("Loading AMazon_digital_Music data set...")
            data = []
            filePath = './Data/Adigital_music/AMazon_digital_Music.dat'
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
        elif fileName == 'aele':
            print("Loading AMazon_Electronics data set...")
            data = []
            filePath = './Data/Aelectronics/AMazon_Electronics.dat'
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
        elif fileName == 'a_hak':
            print("Loading AMazon_home_and_kitchen data set...")
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
        elif fileName == 'akindle':
            print("Loading AMazon_kindle_store data set...")
            data = []
            filePath = './Data/Akindle_store/AMazon_kindle_store.dat'
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
        elif fileName == 'amovie_tv':
            print("Loading AMazon_movies_and_TV data set...")
            data = []
            filePath = './Data/Amovies_and_TV/AMazon_movies_and_TV.dat'
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

    def getAttr(self):
        userAttr = getUserFeature()
        movieAttr = getGenre()

        userAttrKeys = list(userAttr.keys())
        movieAttrKeys = list(movieAttr.keys())
        self.userAttrD = len(userAttr[userAttrKeys[0]])
        self.movieAttrD = len(movieAttr[movieAttrKeys[0]])

        # 扩展自编码器的输入
        userFeatAttr = np.zeros(shape=(self.shape[0], self.userAttrD))
        movieFeatAttr = np.zeros(shape=(self.shape[1], self.movieAttrD))
        for userId in userAttr.keys():
            userFeatAttr[userId] = userAttr[userId]
        for movieId in movieAttr.keys():
            movieFeatAttr[movieId] = movieAttr[movieId]

        self.userFeatAttr = userFeatAttr
        self.movieFeatAttr = movieFeatAttr

    def getTrainTest(self):
        data = self.data
        # data = sorted(data, key=lambda x: (x[0], x[3]))
        train = []
        test = []
        userKeyDict = dict()
        for i in range(len(data)):
            user = data[i][0]
            if user not in userKeyDict.keys():
                userKeyDict[user] = []
            userKeyDict[user].append(data[i])

        for userId in userKeyDict.keys():
            userList = userKeyDict[userId]
            if len(userList)==1:
                j = np.random.randint(self.shape[1])
                while j==userList[0][1]-1:
                    j = np.random.randint(self.shape[1])
                test.append((userList[0][0]-1 , j , userList[0][1]-1 , userList[0][2]))
            else:
                userList = sorted(userList , key=lambda x:x[3])
                # print(userList)
                for i in range(len(userList)-2):
                    user = userList[i][0] - 1
                    item = userList[i][1] - 1
                    nextItem = userList[i+1][1] - 1
                    rate = userList[i][2]
                    train.append((user, item , nextItem, rate))
                test.append((userList[-2][0] - 1, userList[-2][1] - 1, userList[-1][1] - 1, userList[-1][2]))

        return train, test

    def getTrainDict(self):
        dataDict = {}
        for i in self.train:
            if i[0] not in dataDict.keys():
                dataDict[i[0]] = []
            dataDict[i[0]].append(i[1])

        for userId in range(self.shape[0]):
            if userId not in dataDict.keys():
                dataDict[userId] = []
        return dataDict
    #
    # def getEmbedding(self):
    #     train_matrix = np.zeros([self.shape[0], self.shape[1]], dtype=np.float32)
    #     for i in self.train:
    #         user = i[0]
    #         movie = i[1]
    #         rating = i[2]
    #         train_matrix[user][movie] = rating
    #     return np.array(train_matrix)

    def getInstances(self, data, negNum):
        user = []
        item = []
        nextItem = []
        rate = []
        for i in data:
            user.append(i[0])
            item.append(i[1])
            nextItem.append(i[2])
            rate.append(1.0)
            for t in range(negNum):
                j = np.random.randint(self.shape[1])
                while j==i[2]:
                    j = np.random.randint(self.shape[1])
                user.append(i[0])
                item.append(i[1])
                nextItem.append(j)
                rate.append(0.0)
        return np.array(user), np.array(item) , np.array(nextItem), np.array(rate)

    def getTestNeg(self, testData, negNum):
        user = []
        item = []
        nextItem = []
        for s in testData:
            tmp_user = []
            tmp_item = []
            tmp_nextItem = []
            u = s[0]
            i = s[1]
            next_i = s[2]
            tmp_user.append(u)
            tmp_item.append(i)
            tmp_nextItem.append(next_i)
            for t in range(negNum):
                j = np.random.randint(self.shape[1])
                while j in self.trainDict[u]:
                    j = np.random.randint(self.shape[1])
                tmp_user.append(u)
                tmp_item.append(i)
                tmp_nextItem.append(j)
            user.append(tmp_user)
            item.append(tmp_item)
            nextItem.append(tmp_nextItem)
        return [np.array(user), np.array(item), np.array(nextItem)]
