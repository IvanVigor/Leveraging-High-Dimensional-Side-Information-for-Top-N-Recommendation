import math
import numpy as np
import timeit
import heapq

from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from numpy.linalg import inv
from datetime import datetime

# This is an implementation in Python of "Leveraging High-Dimensional Side Information for Top-N Recommendation" published by
# Yifan Chen and Xiang Zhao. PDF and references https://arxiv.org/abs/1702.01516

class HSlimLF:
    def __init__(self, beta = 0.005, learningRate = 1, lambda1 = 0.01, lambda2 = 0.01, threshold = 0.0004, test_boolean = True, split_percentage = 0.75): #beta = 0.01
        super(HSlimLF, self).__init__()
        self.beta = beta
        self.learningRate = learningRate
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.threshold = threshold
        self.test_boolean = test_boolean
        self.split_percentage = split_percentage

    print("Start Time:   " + str(datetime.now()))
    start_time = timeit.default_timer()


    def IDF(self, R):  # Inverted document frequency
        vector_w = []
        for x in range(R.shape[1]):
            features = np.sum(R[:,x].astype(bool))
            vector_w.append(math.log2(R.shape[0] / features))
        return vector_w

    def update_matrix_S(self, F):  # consine_similarity function declaration
        return cosine_similarity(F)

    def update_matrix_S1(self, matrix_r, matrix_v1, matrix_Sw):   # calculate according to Equation (4)
        prod = np.dot(matrix_r.T, matrix_r)
        sub = np.subtract(self.beta * matrix_Sw, matrix_v1)
        return np.dot(inv(np.add(prod, np.dot(self.beta, np.identity(matrix_r.shape[1])))),
                      np.add(prod, sub))

    def shrinkLambda(self, x):  # operator shrinkλ(x) = max{|x|−λ,0}·sgn(x) used to calculate Equation (5)
        return (np.sign(x) * np.max(abs(x) - self.lambda1, 0))

    def update_matrix_S2(self, matrix_Sw, matrix_v2): # calculated according Equation (5)
        combined = np.subtract(matrix_Sw, np.divide(matrix_v2, self.beta))
        for x in range(len(combined)):  # evalution element wise of matrix_Sw
            for y in range(len(combined[0])):
                combined[x, y] = self.shrinkLambda(combined[x, y])
        return combined

    def update_matrix_V1(self, matrix_v1, matrix_s1, matrix_Sw): #update of V1 matrix
        return np.subtract(matrix_v1, np.dot(np.subtract(matrix_s1, matrix_Sw), self.beta))

    def update_matrix_V2(self, matrix_v2, matrix_s2, matrix_Sw): #update of V2 matrix
        return np.subtract(matrix_v2, np.dot(np.subtract(matrix_s2, matrix_Sw), self.beta))

    def calculateQ(self, beta, s1, s2, sw, v1, v2): # calculated according the Equation (3)
        b = np.add(np.subtract(s1,sw),v1 / beta)
        d = np.add(np.subtract(s2, sw), v2 / beta)
        return np.subtract(-beta * b,beta * d)

    def calculateD(self, q, s): # used for computation of D inside the Equation(2)
        d = np.zeros((q.shape[0], q.shape[0]))
        for i in range(q.shape[0]):
            d[i, i] = q[i].T.dot(s[i])
        return d

    def calculateUpdateW(self, pk, q, d, lambda2):  # function expressed by Equation (2) inside the paper
        return pk.T.dot(q - d).dot(pk) + lambda2

    def calculateP_feature(self, f, w, k):
        w = np.sqrt(w)
        fprimo = f.multiply(w)
        d = np.linalg.norm(fprimo, axis=1)
        p = (np.array(fprimo[:,k].T)/d)
        return sparse.csr_matrix(p)

    def calculateP(self, f, w):
        w = np.sqrt(w)
        fprimo = f.todense()
        fprimo = np.multiply(fprimo, w[:, np.newaxis].T)#fprimo = fprimo * w
        d = np.linalg.norm(fprimo, axis=1)
        p = fprimo / d[:,None]
        return p

    def calculateError(self, R,S,w,lambda1,lambda2):
        err = R - R.dot(S)
        return 0.5 * (np.linalg.norm(err)**2) + lambda1 * np.linalg.norm(S, 1) + lambda2 * np.linalg.norm(w, 1)

    def trainWieghtSideInformation(self, num_epocs,matrix_F,matrix_R):

        vector_W = self.IDF(matrix_F) # inverted document frequency - (line 1 Algorithm)
        matrix_P = self.calculateP(matrix_F, vector_W)
        matrix_P = sparse.csr_matrix(matrix_P)
        matrix_SW = self.update_matrix_S(matrix_P)  #calculate according to Equation(1) - (line 2 Algorithm)
        matrix_S1 = matrix_SW   # line 3 Algorithm
        matrix_S2 = matrix_SW   # line 3 Algorithm
        matrix_V1 = np.zeros(matrix_S1.shape)  # initialization to 0 - line 3 Algorithm
        matrix_V2 = np.zeros(matrix_S1.shape)  # initialization to 0 - line 3 Algorithm

        #matrix_P = matrix_P.todense()
        for i in range(num_epocs):  # for each weight
            for k in range(len(vector_W)):      #for each weight
                print(k)  # until it converges
                alpha = self.learningRate
                while (True):
                    print("and the error is...." + str(
                        self.calculateError(matrix_R, matrix_SW, vector_W, self.lambda1, self.lambda2)))
                    # print(timeit.default_timer() - start_time)
                    q = self.calculateQ(self.beta, matrix_S1, matrix_S2, matrix_SW, matrix_V1, matrix_V2)
                    d = self.calculateD(q, matrix_SW)
                    deltaW = self.calculateUpdateW(matrix_P[:, k].toarray().flatten(), q, d, self.lambda2)
                    newWeight = (vector_W[k] - alpha * deltaW)
                    alpha = alpha * 0.95
                    if (newWeight >= 0):
                        vector_W[k] = newWeight
                    else:
                        vector_W[k] = 0
                    matrix_P[:, k] = self.calculateP_feature(matrix_F, vector_W, k).T
                    matrix_SW = self.update_matrix_S(matrix_P)
                    print(vector_W[k])
                    if (math.fabs(alpha * deltaW) < self.threshold or newWeight <= 0):  # do-while
                        break
            matrix_S1 = self.update_matrix_S1(matrix_R, matrix_V1, matrix_SW)
            matrix_S2 = self.update_matrix_S2(matrix_SW, matrix_V2)
            matrix_V1 = self.update_matrix_V1(matrix_V1, matrix_S1, matrix_SW)
            matrix_V2 = self.update_matrix_V1(matrix_V2, matrix_S2, matrix_SW)
        return vector_W



