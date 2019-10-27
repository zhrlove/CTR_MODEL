import csv
import pickle
import pandas as pd
import numpy as np

path = "movies.dat"

def getGenre():
    f = open(path ,'r')
    count = 0
    genre_set = set()
    genre_vectors = {}
    for line in f:
        l = line.split("::")
        count += 1
        genre = l[2].strip().split('|')
        for i in range(len(genre)):
            if genre[i] != '':
                genre_set.add(genre[i])
    genre_result = list(genre_set)
    genre_count = len(genre_set)
    f.close()
    f = open(path,'r')
    for line in f:
        temp = [0] * genre_count
        l = line.split("::")
        genre = l[2].strip().split('|')
        for i in range(len(genre)):
            if genre[i] != '':
                temp[genre_result.index(genre[i])] = 1
        genre_vectors[int(l[0]) - 1] = temp
    f.close()
    return genre_vectors