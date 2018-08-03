from collections import OrderedDict
from itertools import product
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import cPickle


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions ^ test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy

def grid_search(X_train, X_test, y_train, y_test):
    n_estimators = [int(x) for x in np.linspace(start = 50, stop = 2000, num = 20)]
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

    # Create a base model
    rf = RandomForestClassifier(random_state=42)

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid=param_grid, cv=10, n_jobs=-1, verbose=2, return_train_score=True)

    #Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    grid_best_params = grid_search.best_params_
    print grid_best_params

    #base model
    base_model = RandomForestClassifier(random_state=42)
    base_model.fit(X_train, y_train)
    base_accuracy = evaluate(base_model, X_test, y_test)

    #Evaluate best model from the grid search
    best_grid = grid_search.best_estimator_
    grid_accuracy = evaluate(best_grid, X_test, y_test)
    print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))

    #save model
    cPickle.dump(best_grid, './RF_models/RF_model_%(bootstrap)s_%(max_depth)s_%(max_features)s_%(min_samples_leaf)s_%(min_samples_split)s_%(n_estimators)s.pkl' % grid_best_params)

    return best_grid
