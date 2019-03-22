import numpy as np
from time import time

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans, AgglomerativeClustering

# Normalized mutual information is only available
# in the current development version. See if we can import,
# otherwise use dummy.

from sklearn.metrics import normalized_mutual_info_score

from tree_entropy import tree_information
from itm import ITM
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import warnings

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def do_experiments(dataset):
    X, y = dataset.data, dataset.target
    dataset_name = dataset.DESCR.split('\n')[0]
    if dataset_name.startswith("Iris"):
        # iris has duplicate data points. That messes up our
        # MeanNN implementation.
        from scipy.spatial.distance import pdist, squareform
        dist = squareform(pdist(X))
        doubles = np.unique(np.where(np.tril(dist - 1, -1) == -1)[0])
        mask = np.ones(X.shape[0], dtype=np.bool)
        mask[doubles] = False
        X = X[mask]
        y = y[mask]

    n_clusters = len(np.unique(y))
    print("\n\nDataset %s samples: %d, features: %d, clusters: %d" %
          (dataset_name, X.shape[0], X.shape[1], n_clusters))
    print("=" * 70)

    classes = [ITM(n_clusters=n_clusters, infer_dimensionality=False),
               ITM(n_clusters=n_clusters, infer_dimensionality=True),
          #     AgglomerativeClustering(linkage='ward', n_clusters=n_clusters),
               KMeans(n_clusters=n_clusters)]
    names = ["ITM", "ITM ID", "KMeans"]
    for clusterer, method in zip(classes, names):
        start = time()
        clusterer.fit(X)
        y_pred = clusterer.labels_

        ari = adjusted_rand_score(y, y_pred)
        ami = adjusted_mutual_info_score(y, y_pred)
        nmi = normalized_mutual_info_score(y, y_pred)
        objective = tree_information(X, y_pred)

        runtime = time() - start

        print("%-15s ARI: %.3f, AMI: %.3f, NMI: %.3f objective: %.3f time:"
              "%.2f" % (method, ari, ami, nmi, objective, runtime))
        
        # confusion matrix, compare actual y and y predict 
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y, y_pred)
        np.set_printoptions(precision=2)

        # Plot confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes = sorted(np.unique(y)),
                      title='Confusion Matrix for '+method)
        plt.savefig('Confusion Matrix for '+ dataset_name + ' '+ method)
        plt.show()
            
    i_gt = tree_information(X, y)
    print("GT objective: %.3f" % i_gt)


if __name__ == "__main__":
    from sklearn import datasets
    # two datasets: iris and digits
    iris = datasets.load_iris()
    digits = datasets.load_digits()

    dataset_list = [iris, digits]  
    # dataset_list = [mnist]
    for dataset in dataset_list:
        do_experiments(dataset)
