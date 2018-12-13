import numpy as np
import scipy.io
from scipy import sparse
from Hslim import HSlimLF
import Slim
import heapq

def createMatrixInteraction(apapers,numItems, test_boolean, split_value):
    matrix_Interactions = np.zeros((apapers.shape[0], numItems), dtype=bool)
    matrix_Interactions_Test = None
    if(test_boolean == True):
        matrix_Interactions_Test = np.zeros((apapers.shape[0], numItems), dtype=bool)
    k = 0
    for x in range(0, apapers.shape[0]):
        for y in apapers[x][0][0]:
            if(test_boolean == True and k%split_value==0):
                matrix_Interactions_Test[x,y-1]= True       #matlab notation starts from 1
            else:
                matrix_Interactions[x, y-1] = True
            k+=1
    return matrix_Interactions, matrix_Interactions_Test

def MAP5(listout, matrixinteractions, matrixinteractionsTest):  # MAP@5 Used to evaluate the performances
    tot = 0
    cont = 0
    for x in listout:
        single = 0
        utente = x.split(",")
        list = []
        valori = utente[1].split(" ")
        if (matrixinteractionsTest[int(utente[0]), int(valori[0])] == True):
            list.append(True)
        else:
            list.append(False)
        if (matrixinteractionsTest[int(utente[0]), int(valori[1])] == True):
            list.append(True)
        else:
            list.append(False)
        if (matrixinteractionsTest[int(utente[0]), int(valori[2])] == True):
            list.append(True)
        else:
            list.append(False)
        if (matrixinteractionsTest[int(utente[0]), int(valori[3])] == True):
            list.append(True)
        else:
            list.append(False)
        if (matrixinteractionsTest[int(utente[0]), int(valori[4])] == True):
            list.append(True)
        else:
            list.append(False)
        for x in range(5):
            if (list[x] == True):
                for y in range(x + 1):
                    if (list[y] == True):
                        single += 1 / (x + 1)

        presenza = (np.sum(matrixinteractions[int(utente[0]), :].astype(bool).toarray().flatten()))

        if (presenza != 0):
            single = single / min(5, presenza)
            cont += 1
            tot += single
    tot /= cont
    return tot

def makeRecommendation(matrixSimilarity, matrixinteractionsSparse, n):  # Evaluate product between the items and its similarity
    listValue = []
    for index in range(matrixinteractionsSparse.shape[0]):
        scores = matrixinteractionsSparse[index, :].dot(matrixSimilarity)  # calculate score for each item
        scores = scores[0].toarray()[0]
        scores *= np.negative(
            (matrixinteractionsSparse[index, :]).astype(bool).toarray()[0])  # remove already interacted items
        topItems = heapq.nlargest(n, range(len(scores)), scores.take)  # get top N items
        listValue.append(
            str(index) + "," + str(topItems[0]) + " " + str(topItems[1]) + " " + str(topItems[2]) + " " + str(
                topItems[3]) + " " + str(topItems[4]))
        print(index)
    return listValue

mat = scipy.io.loadmat('dataset.mat')
B = mat['counts'].T
A, test_Matrix = createMatrixInteraction(mat['apapers'],mat['Np'][0][0], True, 4)   #load matrix interaction and create test set
hslim = HSlimLF()                                                                   #initializa trainer
weightsSideInformation = hslim.trainWieghtSideInformation(1,B,A)                    #train
B = sparse.csr_matrix(np.array(B.todense())*weightsSideInformation)
W = Slim.sslimTrainUser(A,B)                                                        #train SSlim
recommendations = makeRecommendation(W,sparse.csr_matrix(A),5)                      #make reccomendations
A = sparse.csr_matrix(A)
test_Matrix = sparse.csr_matrix(test_Matrix)
print(MAP5(recommendations, A, test_Matrix))                                        #evaluate performances