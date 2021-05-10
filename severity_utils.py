import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import scipy
import collections
import sys, os, re
import numpy as np
import sklearn, sklearn.linear_model, sklearn.metrics, sklearn.model_selection
import sklearn.neighbors, sklearn.neural_network, sklearn.svm
import scipy, scipy.stats
import pickle


test_ids = [   0,   14,   15,   16]



def evaluate(data, labels, title, 
             label_name=None, plot=True, groups=None, seed=0, method="nn", target_str="",cache={}, save_weights=None):
    
    X = data
    y = labels.astype(float)
    res = {}
    
    np.random.seed(seed)
    
#     gss = sklearn.model_selection.GroupShuffleSplit(train_size=0.8,test_size=0.2, random_state=seed)
#     train_inds, test_inds = next(gss.split(X, y, groups))

    train_ids = list(set(range(len(X)))-set(test_ids))
    
    X_train, y_train = X[train_ids], y.iloc[train_ids]
    X_test, y_test = X[test_ids], y.iloc[test_ids]
    
    if method=="lr":
        model = sklearn.linear_model.LinearRegression()
    elif method=="huber":
        model = sklearn.linear_model.HuberRegressor()
    elif method=="mae":
        # loss = max(0, |y - p| - epsilon)
        model = sklearn.linear_model.SGDRegressor(loss="epsilon_insensitive", epsilon=0, l1_ratio=1, random_state=seed)
    elif method=="nn":
        model = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=[1000],
                                                    early_stopping=True, 
                                                    solver="adam",
                                                    random_state=seed)    
    elif method=="skip":
        pass
    else:
        raise Exception("Unknown method " + method)
    
    if method != "skip":
        key = str(labels.sum()) + str(X_train.sum()) + str(model) + str(train_ids) + str(test_ids)
        #print(key)
        if not (key in cache):
            cache[key] = model.fit(X_train, y_train)
        model = cache[key]
        
        
    if save_weights:
        print("Writing weights to {}".format(save_weights))
        pickle.dump((model.coefs_,model.intercepts_), open(save_weights,"bw"))
    
    ##########
    if method=="skip":
        print("skip - no training, just running evaluation")
        y_pred = X_test
    else:            
        y_pred = model.predict(X_test)
           
    
    res["name"] = title
    res["R^2"] = sklearn.metrics.r2_score(y_test, y_pred)
    res["Correlation"] = scipy.stats.spearmanr(y_pred,y_test)[0]
    
    abs_diff = np.abs((y_test - y_pred))
    res["MAE"] = abs_diff.mean()
    res["MAE_STDEV"] = abs_diff.std()
    res["MSE"] = sklearn.metrics.mean_squared_error(y_test, y_pred)
    res["label_name"] = label_name
    res["method"] = method
    res["absolute_error"] = np.abs(np.array(y_test - y_pred))

    
    if plot:
        
        fig, ax = plt.subplots(figsize=(6,4), dpi=120)
        for x,y,yp in zip(y_test,y_test,y_pred):
            plt.plot((x,x),(y,yp),color='red',marker='')

        #pmax = int(np.max([y_pred.max(), y_test.max()]))+2
        pmax = 10
        plt.plot(range(pmax),range(pmax), c="gray", linestyle="--")
        plt.xlim(-0.5,pmax-1)
        plt.ylim(-0.5,pmax-1)

        plt.scatter(y_test, y_pred, alpha=0.3);
        plt.ylabel("Model prediction ($y_{pred}$)")
        plt.xlabel("Ground Truth ($y_{true}$)")
        plt.title(title);
        plt.text(0.01,0.97, 
                 "$MAE$={0:0.2f}".format(res["MAE"])+ "\n"
                 , ha='left', va='top', transform=ax.transAxes)
        plt.show()

    return res

