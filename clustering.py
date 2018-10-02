# Basic clustering algorithm

import csv
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
from sklearn.cluster import KMeans

# Get data from a csv file
def readcsv(filename):
    matrix = []
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            matrix += [row]
    return np.array(matrix).astype(float)

# Get data by generating it
# Make two clusters for now
def makedata():
    matrix = []
    for i in range(2):
        numpoints = random.randint(5,20)
        offsetx = random.uniform(5, 20) * (math.pow(-1, random.randint(0,2)))
        offsety = random.uniform(5, 20) * (math.pow(-1, random.randint(0,2)))
        constant = random.uniform(0, 20) * (math.pow(-1, random.randint(0,2)))
        for j in range(numpoints):
            x = random.random() + offsetx
            y = random.random() + offsety
            c = random.random() + constant
            multiplier = random.uniform(0,10)
            row = list(map(lambda x: x * multiplier, [x, y, c]))
            matrix += [row]
    return np.array(matrix)

# TODO - generalize for number of independent random variables
def makedataNum(numclusters):
    matrix = []
    for i in range(numclusters):
        numpoints = random.randint(5,20)
        offsetx = random.uniform(5, 20) * (math.pow(-1, random.randint(0,2)))
        offsety = random.uniform(5, 20) * (math.pow(-1, random.randint(0,2)))
        constant = random.uniform(0, 20) * (math.pow(-1, random.randint(0,2)))
        for j in range(numpoints):
            x = random.random() + offsetx
            y = random.random() + offsety
            c = random.random() + constant
            multiplier = random.uniform(0,10)
            row = list(map(lambda x: x * multiplier, [x, y, c]))
            matrix += [row]
    return np.array(matrix)

# Normalize each row of the matrix
def normalize(matrix):
    normMatrix = []
    for i in range(len(matrix)):
        vector = np.array(matrix[i][:(len(matrix[i]) - 1)])
        norm = np.linalg.norm(vector)        
        normVector = matrix[i] / norm
        normMatrix += [normVector]
    return np.array(normMatrix)

# Group clusters based on values
# For now, let number of independent variables
# be the number of clusters we desire
def kcluster(matrix):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(matrix)
    return kmeans.cluster_centers_

# When system is not overdetermined,
# try to find solution
def solution(matrix):
    coefficients = [matrix[0][:len(matrix[0])-1], matrix[1][:len(matrix[1])-1]]
    consts = [matrix[0][len(matrix[0])-1], matrix[1][len(matrix[1])-1]]
    return np.linalg.solve(coefficients, consts)

def graph():
    output = []
    for _ in range(1000):
        a = 20
        b = 30
        c = 10
        epsilon1 = np.random.normal(scale = 1)
        epsilon2 = np.random.normal(scale = 1)
        output.append([a, b + epsilon2, c + epsilon1])
    fig, ax = plt.subplots()
    ax.set_xlim([-20, 20])
    ax.set_ylim([-20,20])
    for i in range(len(output)):
        start, end = output[i][0:2]
        ax.annotate('', xy = end, xytext = start, arrowprops = dict(facecolor ='red', width = 1))
    plt.show()

# if __name__ == "__main__":
    # data = readcsv("testdata.csv")
    # data = makedata()
    # print(data)
    # data = normalize(data)
    # # print(data)
    # cluster = kcluster(data)
    # # print(cluster)
    # print(solution(cluster))