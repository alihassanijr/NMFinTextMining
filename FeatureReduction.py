import numpy as np
from SparseNMF import SparseNMF


class TermDocumentReduce:
    def __init__(self, n_components=25):
        self.n_components = n_components
        self.nmf_inst = SparseNMF(self.n_components,  tol=1e-9, max_iter=300)

        self.W = None
        self.H = None
        self.cluster_assignments = None

    def fit(self, X):
        self.W = self.nmf_inst.fit_transform(X)
        self.H = self.nmf_inst.components_
        self.cluster_assignments = np.argmax(self.H, axis=0)

    def fit_transform(self, X):
        if self.cluster_assignments is None:
            self.fit(X)
        return self.transform(X)

    def transform(self, X):
        assert self.cluster_assignments is not None
        x_new = np.zeros((X.shape[0], self.n_components))
        for c in range(self.n_components):
            terms = np.where(self.cluster_assignments == c)[0]
            x_new[:, c] = np.linalg.norm(X[:, terms], axis=1)
        return x_new
