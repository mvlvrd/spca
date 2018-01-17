import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin, BaseEstimator

class SPCA(BaseEstimator, RegressorMixin):

    def __init__(self, threshold=0., n_components=1):
        self.threshold = threshold
        self.n_components = n_components

        self._pca = None
        self._regressor = None

        self.corr_ = None
        self.mask_ = None
        self.components_ = None
        

    def __str__(self):
        return str(" ".join([str(_) for _ in [self.threshold, self.n_components, self.mask_.sum()]]))


    def _selectFeatures(self, X, y):
        self.corr_ = np.apply_along_axis(lambda x: np.corrcoef(x,y)[0,1], axis=0, arr=X)
        self.mask_ = abs(self.corr_) > self.threshold
        if self.n_components == 0:
            self.n_components = self.mask_.sum()
        elif not (0 < self.n_components <= self.mask_.sum()):
            #TODO: Be more clear in the warning message.
            raise ValueError("Problems with number of components. Max corr:{}".format(max(abs(self.corr_))))


    def fit(self, X, y):
        self._selectFeatures(X, y)

        self._pca = PCA(self.n_components)

        redX = self._pca.fit_transform(np.array(X[:, self.mask_]))
        self.components_ = self._pca.components_
        
        self._regressor = self._regressor or LinearRegression()
        self._regressor.fit(redX, y)

        return self


    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


    def transform(self, X):
        #if not self._pca:
        #    return self.fit_transform(X)
        #else:
        return self._pca.transform(X[:,self.mask_])


    def score(self, X, y):
        return self._regressor.score(self.transform(X), y)


    def predict(self, X):
        return self._regressor.predict(self.transform(X))


    def _error(self, X, y):
        ypred = self.predict(X)
        return np.sum((y - ypred)**2)



if __name__=="__main__":

    from sklearn.model_selection import GridSearchCV, train_test_split
    from bairData import easyData

    nGenes = 5000
    nSamples = 100
    nTest = 100

    X, y = easyData(nGenes, nSamples)

    grid = {"threshold":np.linspace(0.1,1.,10), "n_components":np.arange(1,5)}
    searcher = GridSearchCV(SPCA(), param_grid=grid, error_score=np.NaN, n_jobs=-1)
    searcher.fit(X,y)
    print(searcher.best_params_, searcher.best_score_)
