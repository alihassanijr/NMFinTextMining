import numpy as np
from sklearn import preprocessing


def purity_score(y_true, y_pred):
    # Encoding the true labels, just to be on the safe side
    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y_true)

    # Calculate purity score
    num_class = len(np.unique(y_true))
    num_clusters = len(np.unique(y_pred))
    lbl = np.unique(y_pred)
    scores = np.zeros((num_class, num_clusters))
    for i in range(0, len(y)):
        scores[y[i], np.where(lbl == y_pred[i])[0]] += 1
    acc = np.sum(np.max(scores, axis=0))
    acc /= len(y)
    return acc
