from collections import OrderedDict
from itertools import product
import pandas as pd
import numpy as np
from scipy.stats import randint
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score
from time import time
import cPickle


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    trues = list(np.hstack(test_labels))
    accuracy = 100 - accuracy_score(predictions, trues)
    precision = 100 - precision_score(predictions, trues)
    recall = 100 - recall_score(predictions, trues)
    print('Model Performance')
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    print('Precision = {:0.2f}%'.format(precision))
    print('Recall = {:0.2f}%'.format(recall))

    return accuracy

def grid_search(base_model, X_train, X_test, y_train, y_test, modeltype, k, n_iters, n_estm_grid, n_max_depth):
    if modeltype.lower()=='default':
        print "Evaluating base model..."
        base_model.fit(X_train, y_train)
        base_accuracy = evaluate(base_model, X_test, y_test)

        return base_model

    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    #random grid
    param_dist = {
        'n_estimators': randint(50, 2000),
        'max_features': max_features,
        'max_depth': randint(10, 110),
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap
    }

    # Instantiate the random search model
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=57)
    rand_search = RandomizedSearchCV(estimator=base_model, param_distributions=param_dist, n_iter=n_iters, cv=skf, verbose=2, random_state=42, n_jobs=-1)#randomized

    start = time()
    #Fit the grid search to the data
    rand_search.fit(X_train, y_train)
    print "Time to complete:", time()-start

    rand_best_params = rand_search.best_params_
    print rand_best_params


    #base model
    print "\nEvaluating base model..."
    base_model.fit(X_train, y_train)
    base_accuracy = evaluate(base_model, X_test, y_test)

    #Evaluate best model from the grid search
    print "\nEvaluating best model from random search..."
    best_rand = rand_search.best_estimator_
    random_accuracy = evaluate(best_rand, X_test, y_test)
    randbase_acc_improvement = 100 * (random_accuracy - base_accuracy) / base_accuracy
    print('Improvement of {:0.2f}%.'.format( randbase_acc_improvement))

    if modeltype.lower()=='random':
        if randbase_acc_improvement <= 0:
            return base_model
        return best_rand

    #get results from the random search to build the parameter grid
    cv_results = rand_search.cv_results_
    zipped_results = zip(cv_results['mean_test_score'], cv_results['param_n_estimators'], cv_results['param_max_features'], cv_results['param_max_depth'], cv_results['param_min_samples_split'], cv_results['param_min_samples_leaf'], cv_results['param_bootstrap'])
    zipped_results_sorted = sorted(zipped_results, key=lambda tup:tup[0], reverse=True)[0:10] #sort and select top 10 candidates

    # grid search after doing random. select top choices...
    param_grid = {'n_estimators': [int(x) for x in np.linspace(start=min(zipped_results_sorted, key=lambda t:t[1])[1], stop=max(zipped_results_sorted, key=lambda t:t[1])[1], num=n_estm_grid)],
                  'max_depth': [int(x) for x in np.linspace(start=min(zipped_results_sorted, key=lambda t:t[3])[3], stop=max(zipped_results_sorted, key=lambda t:t[3])[3], num=n_max_depth)]
                  }

    #random forest model with best max_feature, min_sample, min_samples_leaf, and bootstrap parameter from the top candidate in RandomSearchCV **Note: you can change this and include to grid instead, but it will increase number of tests
    rf_grid = RandomForestClassifier(max_features=zipped_results_sorted[0][2], min_samples_split=zipped_results_sorted[0][4], min_samples_leaf=zipped_results_sorted[0][5], bootstrap=zipped_results_sorted[0][6], random_state=42)

    grid_search = GridSearchCV(estimator=rf_grid, param_grid=param_grid, cv=skf, n_jobs=-1, verbose=2, return_train_score=True)

    start = time()
    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    print "Time to complete:", time() - start

    grid_best_params = grid_search.best_params_
    print grid_best_params

    # Evaluate best model from the grid search
    print "\nEvaluating best model from grid search..."
    best_grid = grid_search.best_estimator_
    grid_accuracy = evaluate(best_grid, X_test, y_test)
    gridbase_acc_improvement = 100 * (grid_accuracy - base_accuracy) / base_accuracy
    print('Improvement of {:0.2f}%.'.format(gridbase_acc_improvement))

    if modeltype.lower()=='grid':
        if (gridbase_acc_improvement) <= 0: # if grid search is worse than base model
            if randbase_acc_improvement <= 0: # if random search is worse than base model
                return base_model
            else:
                return best_rand
            return base_model
        return best_grid

    return best_grid
