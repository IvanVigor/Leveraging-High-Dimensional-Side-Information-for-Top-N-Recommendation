import numpy as np
import scipy.io

from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.linear_model import ElasticNet,SGDRegressor
from scipy.sparse import vstack
from scipy.sparse import lil_matrix

"""
cSLIM basic implementation. To understand deeply how it works we encourage you to
read "Sparse Linear Methods with Side Information for Top-N Recommendations"
"""
def sslimTrainUser(A, B, l1_reg=0.001, l2_reg=0.0001):
    listTopSimilar = []

    alpha = l1_reg + l2_reg
    l1_ratio = l1_reg / alpha


    # Following cSLIM proposal on creating an M' matrix = [ M, FT]
    # * alpha is used to control relative importance of the side information
    A = sparse.csc_matrix(A)
    matrixinteractionsSparseNorm = normalize(A, norm='l2', axis=0)
    matrixSimilarity = matrixinteractionsSparseNorm.T.dot(matrixinteractionsSparseNorm)
    #np.sqrt(alpha) * B
    Balpha = B
    #Mline = matrixSimilarity
    Mline = vstack((A, Balpha.T), format='lil')
    m, n = A.shape
    # Fit each column of W separately
    Mline = Mline.tocsc()
    matrixSimilarity = matrixSimilarity.tocsc()
    columns = A.shape[1]

      # sort by column
    for i in range(columns):
        if(i%1000 == 0):
            print(i)
        minimum = min(100,matrixSimilarity[:,i].nnz)            #prendo minimo tra 100 e il numero di item simili
        #print(minimum)
        #top_k_idx = np.argpartition(matrixSimilarity[i,:], -maximum)[:maximum]
        top_k_idx = matrixSimilarity[:, i].data.argpartition(-minimum)[-minimum:]
        listTopSimilar.append(matrixSimilarity[:, i].indices[top_k_idx])

    model = ElasticNet(alpha=0.0001, positive=True, l1_ratio=0.0001, fit_intercept=False, copy_X=False)
    values, rows, cols = [], [], []
    values2, rows2, cols2 = [], [], []
    for j in range(columns):
        # get the target column
        if (len(listTopSimilar[j]>0)):#Mline[:, j].nnz > 0):   #
            if (j % 100 == 0):
                print(j)
            y = Mline[:, j].toarray().flatten()
            # y = column_or_1d(y, warn=True)
            # set the j-th column of X to zero
            startptr = Mline.indptr[j]
            endptr = Mline.indptr[j + 1]
            bak = Mline.data[startptr: endptr].copy()
            Mline.data[startptr: endptr] = 0.0
            # fit one ElasticNet model per column
            model.fit(Mline[:, listTopSimilar[j]], y)
            nnz_idx = model.coef_ > 0.0
            if (nnz_idx.sum() > 0):
                values.extend(model.coef_[nnz_idx])
                rows.extend(listTopSimilar[j][nnz_idx].flatten())
                cols.extend(np.ones(nnz_idx.sum()) * j)
                Mline.data[startptr:endptr] = bak
    # generate the sparse weight matrix
    return scipy.sparse.csc_matrix((values, (rows, cols)), shape=(columns, columns), dtype=np.float32)



def sslim_train(A, B, l1_reg=0.0001, l2_reg=0.0001):
    """
    Computes W matrix of SLIM
    This link is useful to understand the parameters used:
        http://web.stanford.edu/~hastie/glmnet_matlab/intro.html
        Basically, we are using this:
            Sum( yi - B0 - xTB) + ...
        As:
            Sum( aj - 0 - ATwj) + ...
    Remember that we are wanting to learn wj. If you don't undestand this
    mathematical notation, I suggest you to read section III of:
        http://glaros.dtc.umn.edu/gkhome/slim/overview
    """
    alpha = l1_reg + l2_reg
    l1_ratio = l1_reg / alpha

    model = SGDRegressor(
        penalty='elasticnet',
        fit_intercept=False,
        alpha=alpha,
        l1_ratio=l1_ratio
    )

    # Following cSLIM proposal on creating an M' matrix = [ M, FT]
    # * alpha is used to control relative importance of the side information

    # Balpha = np.sqrt(alpha) * B
    Balpha = B
    Mline = vstack((A, Balpha), format='lil')
    m, n = A.shape
    # Fit each column of W separately
    W = lil_matrix((m, m))

    columns = Mline.shape[1]
    for j in range(columns):
        print(j)
        if j % 50 == 0:
            print
            '-> %2.2f%%' % ((j / float(columns)) * 100)
        print(j)
        print(columns)
        mlinej = Mline[:, j].copy()

        # We need to remove the column j before training
        Mline[:, j] = 0

        model.fit(Mline, mlinej.toarray().ravel())

        # We need to reinstate the matrix
        Mline[:, j] = mlinej

        w = model.coef_

        # Removing negative values because it makes no sense in our approach
        w[w < 0] = 0

        for el in w.nonzero()[0]:
            W[(el, j)] = w[el]

    return W
