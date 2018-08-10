import click
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from gridsearch import grid_search, evaluate
from sklearn.ensemble import RandomForestClassifier
from pathlib2 import Path
import cPickle
import os
import sys


@click.command()
@click.option('-i', type=str, help='Path to directory with features')
@click.option('-o', default='./', type=str, help='Output model_path/model_name.pkl')
@click.option('--modeltype', default='default', type=str, help='specificy default/random/grid to choose hyperparameter tuning method for RF model')
@click.option('--testsize', default=0.20, type=float, help='Specify test size between 0.0 to 1.0 for stratified split. Default is 0.2.')
@click.option('--k', default=3, type=int, help='Choose 3 or more folds for Stratified K Fold. Default is 3-fold.')
@click.option('--n_iters', default=100, type=int, help='Choose number of tests for RandomSearchCV. Default is 100 iterations.')
@click.option('--n_estm_grid', default=10, type=int, help='Choose number of parameters for n_estimators to try for GridSearchCV. Default is 10 parameters for n_estimators.')
@click.option('--n_max_depth', default=10, type=int, help='Choose number of parameters for max_depth to try for GridSearcCV. Default is 10 paramaters for max_depth.')
@click.option('--yes', is_flag=True, help='Overwrites any files')
def main(i, o, modeltype, testsize, k, n_iters, n_estm_grid, n_max_depth, yes):
    if modeltype.lower() != 'default' and modeltype.lower() != 'random' and modeltype.lower() != 'grid':
        print >> sys.stderr, "Please select 'default', 'random', or 'grid' as a parameter. "
        sys.exit(1)

    input_directory = i
    output_path = o

    #get feature paths for X_all, y_all numpy arrays
    input_directory = os.path.join(input_directory, '')
    X_all_paths = glob.glob(input_directory + '*X_all.npy')
    y_all_paths = glob.glob(input_directory + '*y_all.npy')
    X_all_paths.sort(key=lambda x: x.split('/')[-1].replace('_X_all.npy', ''))
    y_all_paths.sort(key=lambda x: x.split('/')[-1].replace('_y_all.npy', ''))
    feature_paths = zip(X_all_paths, y_all_paths)

    # concatenate all of the data
    for i, (X_all_path, y_all_path) in enumerate(feature_paths):
        X_all_next = np.load(X_all_path)
        y_all_next = np.load(y_all_path)
        if i == 0:
            X_all = X_all_next
            y_all = y_all_next
        if i != 0:
            X_all = np.vstack((X_all, X_all_next))
            y_all = np.concatenate([y_all, y_all_next])

    # stratified split sampling
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=testsize, stratify=y_all, random_state=42)

    # Create a base model
    rf = RandomForestClassifier(random_state=42)

    #grid_search

    best_model = grid_search(rf, X_train, X_test, y_train, y_test, modeltype, k, n_iters, n_estm_grid, n_max_depth)

    #make the directories if it doesn't exist
    Path('/'.join(output_path.split('/')[0:-1])).mkdir(parents=True, exist_ok=True)

    #save the model as pickle file
    if os.path.isfile(output_path):
        if yes:
            with open(output_path, 'wb') as pkl_file:
                cPickle.dump(best_model, pkl_file)
        elif click.confirm(output_path + " already exists, are you sure you want to overwrite?"):
            with open(output_path, 'wb') as pkl_file:
                cPickle.dump(best_model, pkl_file)
    else:
        with open(output_path, 'wb') as pkl_file:
            cPickle.dump(best_model, pkl_file)


    return

if __name__ == '__main__':
    main()