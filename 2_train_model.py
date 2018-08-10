import click
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from gridsearch import grid_search, evaluate
from sklearn.ensemble import RandomForestClassifier
from pathlib2 import Path
import cPickle
import os


@click.command()
@click.option('-i', type=str, help='Path to directory with features')
@click.option('-o', default='./', type=str, help='Output model path/name of model')
@click.option('--default', is_flag=True, help='Use only the default RF model')
@click.option('--random', is_flag=True, help='Use Randomized search on hyperparameters')
@click.option('--grid', is_flag=True, help='Use Grid search on hyperparameters')
@click.option('--yes', is_flag=True, help='Overwrites any files')
def main(i, o, default, random, grid, yes):
    input_directory = i
    output_path = o

    #get feature paths for X_all, y_all numpy arrays
    if input_directory.rfind('/') != len(input_directory)-1:
        input_directory += '/'
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
    # TODO: ask Mike if he would like to specify test_size
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20, stratify=y_all, random_state=42)

    # Create a base model
    rf = RandomForestClassifier(random_state=42)

    #grid_search

    best_model = grid_search(rf, X_train, X_test, y_train, y_test, default, random, grid)

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