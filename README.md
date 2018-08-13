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

### Creating geojsons
Follow my Python Notebook [Creating segmented geojson](https://github.com/aalten77/WildfiresPilot/blob/master/Creating%20segmented%20geojson.ipynb) to see how this is done. You may run through it and try out your own CatalogImage.

This Notebook will give you multiple geojsons, so before heading to next steps, run this with the script Combine_geojsons.py to combine geojsons to one.
```
python Combine_geojsons.py <directory_path_to_geojsons>
```

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
