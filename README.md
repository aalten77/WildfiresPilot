# WildfiresPilot

This project is an extension to GBDX Notebook 	
[Post-Fire Damage Assessment from High Resolution Imagery](https://notebooks.geobigdata.io/hub/tutorials/5b47cfb82486966ea89b75fd?tab=code). This project will provide the following:
1. Segment image of wildfire and convert to geojson
2. Script to convert geojson to features for RandomForestClassifier
3. Script to train the RandomForestClassifier and optionally use hyperparameter tuning

## Getting Started

To start, you will need to install Anaconda for Python Notebook with GBDXtools virtualenv or you can work in GBDX Notebooks. To find how to set up gbdxtools locally follow this link: https://github.com/GeoBigData/gbdx-training/tree/master/gbdxtools_module

I also recommend the following:
* Python 2.7.15
* Pycharm or preferred IDE
* QGIS or your preferred GIS software

## Creating Geojsons
Follow my Python Notebook [Creating segmented geojson](https://github.com/aalten77/WildfiresPilot/blob/master/Creating%20segmented%20geojson.ipynb) to see how this is done. You may run through it and try out your own CatalogImage.

This Notebook will give you multiple geojsons, so before heading to next steps, run this with the script Combine_geojsons.py to combine geojsons to one.
```
python Combine_geojsons.py <directory_path_to_geojsons>
```
After you have created the geojson, you can optionally view it in QGIS and edit the file to your needs. Relabel polygons as necessary.

## Convert Geojson to Features
This script is used to take the labelled geojson and convert to Numpy array containing the dataset of RSI features. You can also give it a zipped geojson if your remote repository doesn't have capacity for the raw geojson data. The Numpy arrays will be saved out to your specified output directory.
```
python 1_geojson_to_feats.py -i <path_to_geojson> -o <output_directory> --yes
```

For help on the options, run the following:
```
python 1_geojson_to_feats.py --help
```
## Random Forest Model Training
This next script will load the Numpy arrays and train the default [RandomForestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) model from Scikit-learn. Optionally, you can toggle for Randomized search of Grid search for hyperparameter tuning. 

Below is an example of the options to run with the script:
```
python 2_train_model.py -i <directory_path_to_numpy_feats> -o <output_model_path/model_name.pkl> --modeltype 'grid' --testsize 0.3 --k 10 --n_estm_grid 10 --n_max_depth 10 --yes
```

For help on the options, run the following: 
```
python 2_train_model.py --help
```

### Inspiration for Hyperparameter Tuning

The usage of random search and grid search is proposed by William Koehrsen in his article, [Hyperparameter Tuning the Random Forest in Python](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74).

## Authors

* **Ai-Linh Alten** - *Initial work* - [aalten77](https://github.com/aalten77)

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
