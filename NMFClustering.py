import numpy as np
from SparseNMF import SparseNMF


class TermDocumentReduce:
    """
    Reduces the dimensions of the original term-document matrix into a number of components specified, just like LSA.
    The only difference is that LSA is only a dim-reduction method and therefore it tries to keep changes to the space
    at a minimum (using singular value based projections), but this method uses SparseNMF to cluster the terms into
    `topics` and then combines the term vectors (feature vectors) within each topic into a single feature vector, which
    creates a completely new space, which can sometimes be more suitable for clustering.
    """
    def __init__(self, n_components=25):
        self.n_components = n_components
        self.nmf_cluster = NMFCluster(self.n_components)

    def fit(self, X):
        self.nmf_cluster.fit(X)

    def fit_transform(self, X):
        if self.nmf_cluster.topics_ is None:
            self.fit(X)
        return self.transform(X)

    def transform(self, X):
        assert self.nmf_cluster.topics_ is not None
        x_new = np.zeros((X.shape[0], self.n_components))
        for c in range(self.n_components):
            terms = np.where(self.nmf_cluster.topics_ == c)[0]
            x_new[:, c] = np.linalg.norm(X[:, terms], axis=1)
        return x_new


class NMFCluster:
    """
    Clusters the documents and terms in a term-document matrix using SparseNMF.
    """
    def __init__(self, n_clusters=25):
        self.n_clusters = n_clusters
        self.nmf_inst = SparseNMF(n_clusters,  tol=1e-9, max_iter=300)

        self.W = None
        self.H = None
        self.labels_ = None
        self.topics_ = None

    def fit(self, X):
        self.W = self.nmf_inst.fit_transform(X)
        self.H = self.nmf_inst.components_
        self.labels_ = np.argmax(self.W, axis=1)
        self.topics_ = np.argmax(self.H, axis=0)

    def fit_predict(self, X):
        """
        Fits SparseNMF on X and returns the clustering assignments (lables) from argmax
        """
        if self.labels_ is None:
            self.fit(X)
        return self.labels_

    def fit_predict_documents(self, X):
        """
        Fits SparseNMF on X and returns the clustering assignments (lables) from argmax
        """
        return self.fit_predict(X)

    def fit_predict_terms(self, X):
        """
        Fits SparseNMF on X and returns the term clustering assignments (topics) from argmax
        """
        if self.topics_ is None:
            self.fit(X)
        return self.topics_

    def predict(self, X):
        """
        Transforms an input X into an n-by-k matrix and clusters using argmax
        """
        return np.argmax(np.dot(X, self.H.T), axis=1)

    def predict_documents(self, X):
        """
        Transforms an input X into an n-by-k matrix and clusters using argmax
        """
        return self.predict(X)

    def predict_terms(self, X):
        """
        Transforms an input X into an n-by-k matrix and clusters using argmax
        """
        return np.argmax(np.dot(X.T, self.W.T), axis=1)
