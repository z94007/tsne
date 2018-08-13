import numpy as np

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar
def load_data(dataset):
    """

    Load the dataset, perform shuffling

    """
    with gzip.open(dataset, 'rb') as f:
        train_x, train_y = cPickle.load(f)
    if scipy.sparse.issparse(train_x):
        train_x = train_x.toarray()
    if train_x.dtype != 'float32':
        train_x = train_x.astype(np.float32)
    if train_y.dtype != 'int32':
        train_y = train_y.astype(np.int32)

    if train_y.ndim > 1:
        train_y = np.squeeze(train_y)
    N = train_x.shape[0]
    idx = np.random.permutation(N)
    train_x = train_x[idx]
    train_y = train_y[idx]

    return train_x, train_y
    
filename = 'data.pkl.gz'
path = './data/'
dataset = path+filename
(x, y)=load_data(dataset)
(X, y)=(x[:,:15],y)
(x_pca, y_pca)=(x[:,15:],y)

import pandas as pd

feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]

df = pd.DataFrame(X,columns=feat_cols)
df['label'] = y
df['label'] = df['label'].apply(lambda i: str(i))

X, y = None, None

print 'Size of the dataframe: {}'.format(df.shape)

rndperm = np.random.permutation(df.shape[0])

import time

from sklearn.manifold import TSNE

n_sne = 7000

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)

print 't-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start)

from ggplot import *
df_tsne = df.loc[rndperm[:n_sne],:].copy()
df_tsne['x-tsne'] = tsne_results[:,0]
df_tsne['y-tsne'] = tsne_results[:,1]

chart = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label') ) \
        + geom_point(size=70,alpha=0.1) \
        + ggtitle("tSNE dimensions colored by digit")
ggsave(chart,'tsneplot.png')