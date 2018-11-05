from __future__ import division
import numpy as np
import scipy
import graphviz
import matplotlib
from math import floor, ceil
import xgboost as xgb

def main(_):
    
    # creating training data
    data = np.random.rand(5, 10)  # 5 entities, each contains 10 features
    label = np.random.randint(2, size=5)  # binary target
    dtrain = xgb.DMatrix(data, label=label)

    #csr = scipy.sparse.csr_matrix((dat, (row, col))) #  data creation using scipy
    #dtrain = xgb.DMatrix(csr)
    
    # Booster parameters
    param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
    param['nthread'] = 4
    param['eval_metric'] = 'auc'

    # Evaluation parameters
    #param['eval_metric'] = ['auc', 'ams@0']
    plst = param.items()
    plst += [('eval_metric', 'ams@0')]

    # Testing data
    data = np.random.rand(7, 10) # 7 entities, each contains 10 features
    dtest = xgb.DMatrix(data)
    
    # Specify validations set to watch performance
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    
    # Training
    num_round = 10
    bst = xgb.train(plst, dtrain, num_round, evallist)
    bst.save_model('0001.model') # Saving the model
    bst.dump_model('dump.raw.txt') # dump model
    bst.dump_model('dump.raw.txt', 'featmap.txt') # dump model with feature map
    bst = xgb.Booster({'nthread': 4})  # init model
    bst.load_model('model.bin')  # load data
    
    # Testing
    ypred = bst.predict(dtest)
    #ypred = bst.predict(dtest, ntree_limit=bst.best_ntree_limit) # Use this one only if early stopping is enabled in training

    # Plotting
    xgb.plot_importance(bst)
    xgb.plot_tree(bst, num_trees=2)
    xgb.to_graphviz(bst, num_trees=2)

    file = open("results.txt","w")
    file.write(ypred)
    file.close()
