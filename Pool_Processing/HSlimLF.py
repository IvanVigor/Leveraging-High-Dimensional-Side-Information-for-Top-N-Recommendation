import math
import numpy as np
import timeit
import heapq

from sklearn.metrics.pairwise import cosine_similarity


# This is an implementation in Python of "Leveraging High-Dimensional Side Information for Top-N Recommendation" published by
# Yifan Chen and Xiang Zhao. PDF and references https://arxiv.org/abs/1702.01516

class HSlimLF:
    def __init__(self, beta = 0.005, learningRate = 1, lambda1 = 0.01, lambda2 = 0.01, treshold = 0.0004, test_boolean = True, split_percentage = 0.75): #beta = 0.01
        super(HSlimLF, self).__init__()
        self.beta = beta
        self.learningRate = learningRate
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.treshold = treshold
        self.test_boolean = test_boolean
        self.split_percentage = split_percentage

    print("Start Time:   " + str(datetime.now()))
    start_time = timeit.default_timer()

    def trainWieghtSideInformation(self, num_epocs,matrix_F,matrix_R):

        vector_W = self.IDF(matrix_F) # inverted document frequency - (line 1 Algorithm)
        matrix_P = self.calculateP(matrix_F, vector_W)
        matrix_P = sparse.csr_matrix(matrix_P)
        matrix_SW = self.update_matrix_S(matrix_P)  #calculate according to Equation(1) - (line 2 Algorithm)
        matrix_S1 = matrix_SW   # line 3 Algorithm
        matrix_S2 = matrix_SW   # line 3 Algorithm
        matrix_V1 = np.zeros(matrix_S1.shape)  # initialization to 0 - line 3 Algorithm
        matrix_V2 = np.zeros(matrix_S1.shape)  # initialization to 0 - line 3 Algorithm

        matrix_P = matrix_P.todense()
        for i in range(num_epocs):     #for each weight, update
            alpha = self.learningRate
            for k in range(25):#25

                print("and the error is...."+str(self.calculateError(matrix_R,matrix_SW,vector_W,self.lambda1,self.lambda2)))
                q = self.calculateQ(self.beta,matrix_S1,matrix_S2,matrix_SW,matrix_V1,matrix_V2)  # Calculation of Q and D for the computation of ∇w (line 6 - Algorithm)
                d = self.calculateD(q, matrix_SW)
                if(i==0 and k==0):
                    deltaW = np.zeros(len(vector_W))
                    deltaW[0]=0.01
                else:
                    deltaW = self.calculateUpdateW(matrix_P, q, d, self.lambda2,alpha) # calculate according to Equations (2) and (3)
                newWeight = np.subtract(vector_W,deltaW) # P <- [w −ϕ∇w]; (line 8 Algorithm)
                #print(self.calculateError(matrix_R, matrix_SW, newWeight, self.lambda1, self.lambda2) - self.calculateError(matrix_R, matrix_SW, vector_W, self.lambda1, self.lambda2))
                #print(0.01 * (newWeight - vector_W).T.dot(deltaW))
                vector_W = newWeight
                alpha = alpha * 0.95  # heuristically search the step
                for i in range(len(vector_W)):
                    if(vector_W[i]<0): # P value for the update
                        vector_W[i]=0
                matrix_P = self.calculateP(matrix_F, vector_W)
                matrix_SW = self.update_matrix_S(matrix_P)
                tot = 0
                for el in deltaW:
                    tot += math.fabs(el)
                #print(tot/len(vector_W))
            self.beta = self.beta * 0.90
            matrix_S1 = self.update_matrix_S1(matrix_R, matrix_V1, matrix_SW)  # line 10
            matrix_S2 = self.update_matrix_S2(matrix_SW, matrix_V2)            # line 11
            matrix_V1 = self.update_matrix_V1(matrix_V1, matrix_S1, matrix_SW) # line 12
            matrix_V2 = self.update_matrix_V1(matrix_V2, matrix_S2, matrix_SW) # line 13
        return vector_W
