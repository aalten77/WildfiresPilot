from collections import OrderedDict
from itertools import product
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

def grid_search(X_train, X_test, y_train, y_test):
    n_estimators = [int(x) for x in np.linspace(start = 10, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    param_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    param_dist = {
        'n_estimators': randint(50, 2000),
        'max_features': max_features,
        'max_depth': randint(10, 110),
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap
    }

    # Create a base model
    rf = RandomForestClassifier(oob_score=True, random_state=42)

    # Instantiate the grid search model
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=57)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=skf, n_jobs=-1, verbose=2, return_train_score=True)
    #n_iter_search = 1000
    #grid_search = GridSearchCV(estimator=rf, param_grid=param_dist, cv=10, n_jobs=-1, verbose=2, return_train_score=True)
    #grid_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=n_iter_search, cv=10, verbose=2, random_state=42, n_jobs=-1)#randomized

    start = time()
    #Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    print "Time to complete:", time()-start
    #print "RandomizedSearchCV took %.2f seconds for %d candidates"
    #" parameter settings." % ((time() - start), n_iter_search)

    grid_best_params = grid_search.best_params_
    print grid_best_params

    #base model
    print "Evaluating base model..."
    base_model = RandomForestClassifier(random_state=42)
    base_model.fit(X_train, y_train)
    base_accuracy = evaluate(base_model, X_test, y_test)

    #Evaluate best model from the grid search
    print "Evaluating best model from grid/random search..."
    best_grid = grid_search.best_estimator_
    grid_accuracy = evaluate(best_grid, X_test, y_test)
    print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))

    #save model
    # TODO: allow user to name the file, remove parameters from file name
    with open('./RF_models/RF_model_%(bootstrap)s_%(max_depth)s_%(max_features)s_%(min_samples_leaf)s_%(min_samples_split)s_%(n_estimators)s.pkl' % grid_best_params, 'wb') as pkl_file:
        cPickle.dump(best_grid, pkl_file)

    return best_grid
