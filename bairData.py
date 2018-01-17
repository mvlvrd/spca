import numpy as np

def hardData(nFeatures, nEvents):
    X = np.empty((nFeatures,nEvents))
    X[:50,:50] = 3.
    X[:50,50:] = 4.
    X[50:100,:] = 3.5 + 1.5*np.random.binomial(1,0.4)*np.ones(50)
    X[100:200,:] = 3.5 + 0.5*np.random.binomial(1,0.7)*np.ones(100)
    X[200:300,:] = 3.5 - 1.5*np.random.binomial(1,0.3)*np.ones(100)
    X[300:,:] = 3.5
    X += np.random.normal(0.0, 1.0, size=X.size).reshape(X.shape)

    y = X[:50,:].sum(axis=0)/25. + np.random.normal(0.0, 1.5, nEvents)

    #Transpose X to fit the scikit convention: (n_samples, n_features)
    return X.T, y

def easyData(nFeatures, nEvents):
    X = np.empty((nFeatures,nEvents))
    X[:50,:50] = 3.
    X[:50,50:] = 4.
    X[50:,:] = 3.5
    X += np.random.normal(0.0, 1.0, size=X.size).reshape(X.shape)
    
    y = X[:50,:].sum(axis=0)/25. + np.random.normal(0.0, 1.5, nEvents)
    
    #Transpose X to fit the scikit convention: (n_samples, n_features)
    return X.T, y
