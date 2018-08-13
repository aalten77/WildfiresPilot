"""""
    Author: Ai-Linh Alten
    Date created: 7/30/2018
    Date last modified: 8/10/2018
    Python Version: 2.7.15

    This script takes geojson produced by image segmentation filter and creates dataset of remote sensing features.
    command:
        1_geojson_to_feats.py --help
"""

import gbdxtools
from gbdxtools import Interface, CatalogImage
import json
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import shape
import geojson
import numpy as np
import numpy.ma as ma
import collections
from scipy import ndimage as ndi
from skimage import filters, color, exposure
import multiprocessing
from readjsonfromurl import get_json_remote
from readjsonfromlocal import get_json_local
import sys
import os
import click
gbdx = Interface()

rsi_dict = dict(arvi=lambda COAST, B, G, Y, R, RE, NIR1, NIR2: (NIR1 - (R - (B - R))) / (NIR1 + (R - (B - R))),
                dd=lambda COAST, B, G, Y, R, RE, NIR1, NIR2: (2 * NIR1 - R) - (G - B),
                gi2=lambda COAST, B, G, Y, R, RE, NIR1, NIR2: (B * -0.2848 + G * -0.2434 + R * -0.5436 + NIR1 * 0.7243 + NIR2 * 0.0840) * 5,
                gndvi=lambda COAST, B, G, Y, R, RE, NIR1, NIR2: (NIR1 - G)/(NIR1 + G),
                ndre=lambda COAST, B, G, Y, R, RE, NIR1, NIR2: (NIR1 - RE)/(NIR1 + RE),
                ndvi=lambda COAST, B, G, Y, R, RE, NIR1, NIR2: (NIR1 - R)/(NIR1 + R),
                ndvi35=lambda COAST, B, G, Y, R, RE, NIR1, NIR2: (G - R)/(G + R),
                ndvi84=lambda COAST, B, G, Y, R, RE, NIR1, NIR2: (NIR2 - Y)/(NIR2 + Y),
                nirry=lambda COAST, B, G, Y, R, RE, NIR1, NIR2: (NIR1)/(R + Y),
                normnir=lambda COAST, B, G, Y, R, RE, NIR1, NIR2: NIR1/(NIR1 + R + G),
                psri=lambda COAST, B, G, Y, R, RE, NIR1, NIR2: (R - B)/ RE,
                rey=lambda COAST, B, G, Y, R, RE, NIR1, NIR2: (RE - Y)/(RE + Y),
                rvi=lambda COAST, B, G, Y, R, RE, NIR1, NIR2: NIR1/R,
                sa=lambda COAST, B, G, Y, R, RE, NIR1, NIR2: (((Y + R) * 0.35)/2) + ((0.7 * (NIR1 + NIR2))/ 2) - 0.69,
                vi1=lambda COAST, B, G, Y, R, RE, NIR1, NIR2: (10000 * NIR1)/(RE) ** 2,
                vire=lambda COAST, B, G, Y, R, RE, NIR1, NIR2: NIR1/RE,
                br=lambda COAST, B, G, Y, R, RE, NIR1, NIR2: (R/B) * (G/B) * (RE/B) * (NIR1/B),
                gr=lambda COAST, B, G, Y, R, RE, NIR1, NIR2: G/R,
                rr=lambda COAST, B, G, Y, R, RE, NIR1, NIR2:(NIR1/R) * (G/R) * (NIR1/RE),
                wvbi=lambda COAST, B, G, Y, R, RE, NIR1, NIR2: (COAST - RE)/(COAST + RE),
                wvnhfd=lambda COAST, B, G, Y, R, RE, NIR1, NIR2: (RE - COAST)/(RE + COAST),
                evi=lambda COAST, B, G, Y, R, RE, NIR1, NIR2: (2.5 * (NIR2 - R))/(NIR2 + 6 * R - 7.5 * B + 1),
                savi=lambda COAST, B, G, Y, R, RE, NIR1, NIR2: ((1 + 0.5) * (NIR2 - R))/ (NIR2 + R + 0.5),
                msavi=lambda COAST, B, G, Y, R, RE, NIR1, NIR2: (2 * NIR2 + 1 - ((2 * NIR2 + 1) ** 2 - 8 * (NIR2 - R)) ** 0.5)/ 2,
                bai=lambda COAST, B, G, Y, R, RE, NIR1, NIR2: 1.0/((0.1 + R) ** 2 + 0.06 + NIR2),
                rgi=lambda COAST, B, G, Y, R, RE, NIR1, NIR2: R/G,
                bri=lambda COAST, B, G, Y, R, RE, NIR1, NIR2: B/R
                )
def calc_rsi(image):
    """Remote sensing indices for vegetation, built-up, and bare soil.
    Adapted from: https://github.com/GeoBigData/nbfirerisk/blob/master/nbfirerisk/ops.py"""

    # roll axes to conventional row,col,depth
    img = np.rollaxis(image, 0, 3)

    # bands: Coastal(0), Blue(1), Green(2), Yellow(3), Red(4), Red-edge(5), NIR1(6), NIR2(7)) Multispectral
    COAST = img[:, :, 0]
    B = img[:, :, 1]
    G = img[:, :, 2]
    Y = img[:, :, 3]
    R = img[:, :, 4]
    RE = img[:, :, 5]
    NIR1 = img[:, :, 6]
    NIR2 = img[:, :, 7]

    rsi_dict_ordered = collections.OrderedDict(sorted(rsi_dict.items(), key=lambda t: t[0]))

    rsi = np.stack(
        [value(COAST, B, G, Y, R, RE, NIR1, NIR2) for key, value in rsi_dict_ordered.items()],
        axis=2)

    return rsi

def power(image, kernel):
    """Normalize images for better comparison.
    Adapted from: https://github.com/GeoBigData/nbfirerisk/blob/master/nbfirerisk/ops.py
    """
    image = (image - image.mean())/image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap') ** 2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap') ** 2)


def calc_gabors(image, frequency=1, theta_vals=[0, 1, 2, 3]):
    """Calculate gabor.
    Adapted from: https://github.com/GeoBigData/nbfirerisk/blob/master/nbfirerisk/ops.py
    """
    # convert to gray scale
    image = image.astype(np.uint8)
    rgb = np.dstack((image[4], image[2], image[1])) #RGB

    img = exposure.equalize_hist(color.rgb2gray(rgb))
    #img = exposure.equalize_hist(color.rgb2gray(image.rgb(blm=True))) #is blm necessary here?
    results_list = []
    for theta in theta_vals:
        theta = theta / 4. * np.pi
        kernel = filters.gabor_kernel(frequency, theta=theta)
        # Save kernel and the power image for each image
        results_list.append(power(img, kernel))

    gabors = np.rollaxis(np.dstack([results_list]), 0, 3)

    return gabors

def pixels_as_features(image, include_gabors=True):
    """Calculates remote sensing indices and gabor filters(optional).
    Returns image features of image bands, remote sensing indices, and gabor filters.
    Adapted from: https://github.com/GeoBigData/nbfirerisk/blob/master/nbfirerisk/ops.py
    """

    # roll axes to conventional row,col,depth
    img = np.rollaxis(image, 0, 3)
    rsi = calc_rsi(image)
    if include_gabors is True:
        gabors = calc_gabors(image)
        stack = np.dstack([img, rsi, gabors])
    else:
        stack = np.dstack([img, rsi])

    feats = stack.ravel().reshape(stack.shape[0] * stack.shape[1], stack.shape[2])

    return feats

def segment_as_feature(image, medoid=True, normalize=False, include_gabors=False):
    """Calls pixel_as_features() to get remote sensing indices and gabor filters (optional). For every segment, the median pixel is selected with the feature stack.
    If mediod=False, then the mean features of al pixels in segment is selected.
    """
    feats = pixels_as_features(image, include_gabors=include_gabors)
    if medoid is True:
    # NOTE: I added in this new normalize argument to handle the issue with different
    # variables having different scales. this helps ensure that the medoid is not affected
    # by different scales of the input variables, RSI, gabors, etc..
        if normalize is False:
            norms = feats
        else:
            mins = feats.min(axis=0)
            maxs = feats.max(axis=0)
            norms = (feats - mins) / (maxs - mins)

        medians = np.median(norms, axis=0)
        dist = np.linalg.norm(norms - medians, axis=1)
        feat = feats[np.argmin(dist), :]
    else:
        feat = np.mean(feats, axis=0)

    return feat

def geojson_to_polygons(js_):
    """Convert the geojson into Shapely Polygons.
    Adapted from: https://gist.github.com/drmalex07/5a54fc4f1db06a66679e

    :param js_: geojson with segments as Polygons
    :return: list of Shapely Polygons of segments"""

    polys = []
    for i, feat in enumerate(js_['features']):
        o = {
            "coordinates": feat['geometry']['coordinates'],
            "type": feat['geometry']['type']
        }
        s = json.dumps(o)

        # convert to geojson.geometry.Polygon
        g1 = geojson.loads(s)

        # covert to shapely.geometry.polygon.Polygon
        g2 = shape(g1)

        polys.append(g2)

    return polys


def get_segment_masks(image, polys, invert=True):
    """Image masks of the segments. Converted the segment polygons to binary images.
    :param image: CatalogImage
    :param polys: list of segment Shapely Polygons
    :param invert: specify whether or not to invert the image mask
    :returns: a list of image aois given the bounds of the segment and list of segment masks
    """
    image_aoi_segs = []
    seg_masks = []

    for poly in polys:
        bounds = poly.bounds
        image_aoi_seg = image.aoi(bbox=list(bounds))
        segment_mask = geometry_mask([poly], transform=image_aoi_seg.affine,
                                 out_shape=(image_aoi_seg.shape[1], image_aoi_seg.shape[2]), invert=invert)

        image_aoi_segs.append(image_aoi_seg)
        seg_masks.append(segment_mask)

    return image_aoi_segs, seg_masks

def load_cat_image(catalog_id, bbox=None, pan=False):
    """Returns the CatalogImage.
    :param catalog_id: catalog_id that is specified in geojson
    :type catalog_id: String
    :param bbox: can set if needed, but not necessary
    :param pan: usually True for pansharpened image, but can set to False if loading image takes a while
    :return: CatalogImage"""

    if bbox == None:
        return CatalogImage(catalog_id, band_type="MS", pansharpen=pan)
    return CatalogImage(catalog_id, band_type="MS", pansharpen=pan, bbox=bbox)

def clean_data(X_all, y_all):
    """Clean data from any Infs or NaNs. Will delete the row from numpy matrix if found in the row.
    :param X_all: data with remote sensing indices and gabors as filters
    :type X_all: numpy masked array
    :param y_all: data with True (burnt) and False (non-burnt) labels for each segment
    :type y_all: numpy array
    :returns: X_all and y_all. X_all is returned as np.float32"""

    # remove infs
    if np.any(np.any(np.isinf(X_all))):
        index_infs = np.argwhere(np.isinf(X_all))
        index_ls = []
        for i, j in index_infs:
            index_ls.append(i)
        mask_inf = np.isfinite(X_all).all(axis=1)
        X_all = X_all[mask_inf, :]
        y_all = np.delete(y_all, index_ls)

    # remove NaNs
    if np.any(np.any(np.isnan(X_all))):
        index_nans = np.argwhere(np.isnan(X_all))
        index_ls = []
        for i, j in index_nans:
            index_ls.append(i)
        mask_inf = np.isfinite(X_all).all(axis=1)
        X_all = X_all[mask_inf, :]
        y_all = np.delete(y_all, index_ls)

    #set X_all to np.float32 type
    X_all = X_all.astype(np.float32)
    X_all = X_all[0:].astype(np.float32)

    return X_all, y_all

def validate_json(js):
    """Look for incorrect data in the geojson. Like if Tomnod_label is anything other than True/False/None or if the geometry type for every feature isn't a Polygon.
    :param js: geojson
    :returns: geojson and flag. If flag returns 1, then the program will exit.
    """

    flag = -999
    none_type_locs = []
    for i, feat in enumerate(js['features']):
        #check for correct Tomnod_labels. If incorrect, set flag = 1
        if feat['properties']['Tomnod_label'] != True and feat['properties']['Tomnod_label'] != False and \
                feat['properties']['Tomnod_label'] is not None:
            print >> sys.stderr, "ERROR: Incorrect Tomnod_label at feature element {}. Please go fix this.".format(
                i)
            flag = 1

        try:
            #check for geometry type. If incorrect type, set flag = 1
            if feat['geometry']['type'] is not None and feat['geometry']['type'] != "Polygon":
                print >> sys.stderr, "ERROR: Incorrect feature type, must be Polygon only! Discovered at feature element {}. Please go fix this.".format(
                    i)
                flag = 1
        #if NoneType geometries detected, record index locations
        except TypeError:
            print >> sys.stderr, "WARNING: a NoneType for geometry type occured at feature element {}.".format(
                i)
            none_type_locs.append(i)
            continue

    #remove any NoneTypes
    if len(none_type_locs) > 0:
        for index in sorted(none_type_locs, reverse=True):
            del js['features'][index]

    return js, flag

def create_feature_dataset(rsi_samples, js):
    """Create feature stack for X_all and set Tomnod_labels to boolean and store to y_all as a numpy array. Then data is cleaned by clean_data().
    :param rsi_samples:
    :param js: geojson with features
    :returns: X_all and y_all"""

    # entire dataset
    X_all = np.vstack(rsi_samples)
    X_all = X_all.filled() #convert to numpy ndarray
    y_all = [x['properties']['Tomnod_label'] == 1 for x in js['features']] #set labels to boolean instead of int
    y_all = np.array(y_all)
    y_all.reshape(X_all.shape[0], 1)

    # clean data
    X_all, y_all = clean_data(X_all, y_all)

    return X_all, y_all

@click.command()
@click.option('-i', type=str, help='Input geojson path (local or remote)')
@click.option('-o', default='./', type=str, help='Path to output directory')
@click.option('--yes', is_flag=True, help='Overwrites any files')
def main(i, o, yes):
    """This script will convert geojson to features and save as .npy files. You can specify input file path/remote file path for geojson and folder to save the numpy arrays to.
    To look at options for script, run: python 1_geojson_to_feats.py --python
    """
    input_geojson_file = i
    features_directory = o

    print "Read input file path:", input_geojson_file
    if features_directory == './':
        print 'OK, your features data will be stored in the current directory:', features_directory
    else:
        print 'OK, your features data will be stored in:', features_directory


    #fetch geojson from local path or remote path. Throw error if not found.
    print "\nfetching geojson..."
    file_type = 'local' #set default to local anyway assuming not an http/https remote link
    if os.path.exists(input_geojson_file):
        file_type = 'local'
    else:
        if 'http' in input_geojson_file:
            file_type = 'remote'
        else:
            raise ValueError("Could not identify file type as remote or local.")

    if file_type == 'local':
        js = get_json_local(input_geojson_file)
    elif file_type == 'remote':
        js = get_json_remote(input_geojson_file)
        if js == -1:
            print >> sys.stderr, "No remote file found at", input_geojson_file
            sys.exit(1)

    print ""



    # validate json for all geometries and correct labels:
    js, flag = validate_json(js)
    if flag == 1: # if flag is activated then do not execute rest
        sys.exit(1)

    #load CatalogImage given catalog id in geojson
    print "\nloading CatalogImage..."
    image = load_cat_image(js['properties']['catalog_id'], pan=True)

    print "current number of features:", len(js['features']) # total features found in geojson

    # remove any Tomnod labels that are None
    print "\nremoving Tomnod_labels = None..."
    filtered_feats = filter(lambda x: x['properties']['Tomnod_label'] != None, js['features'])
    js['features'] = filtered_feats

    print "new number of features:", len(js['features']) # total features after removing Tomnod labels that are None type

    # load the raster, mask it by the polygon segment
    print "\nmasking image..."
    polys = geojson_to_polygons(js)
    image_aoi_segs, seg_masks = get_segment_masks(image, polys, invert=False) #toggle the inversion if necessary... remember this should be set to True for multiplying image to mask
    image_aoi_blobs = [ma.array(aoi, mask=np.dstack((seg_masks[i],)*8)) for i, aoi in enumerate(image_aoi_segs)] #image masks instead of just multiplying the image to mask. image_aoi_blobs will be a list of numpy masked arrays that mask out 0 (black) pixels


    # TODO: debug masked array in segments/pixels as features, make sure the masked numbers are not playing with the computation
    print "\ncreating features dataset..."
    #creating segments as features -- use asynchronous polling to calculate features for every segment
    p = multiprocessing.Pool(processes=4)
    rsi_samples = [p.apply_async(segment_as_feature, args=(blob,), kwds={'include_gabors': True}) for blob in image_aoi_blobs]
    rsi_samples_output = [p.get() for p in rsi_samples]

    #create X and y dataset
    X_all, y_all = create_feature_dataset(rsi_samples_output, js)

    #save out the dataset
    print "\nsaving the features to", features_directory, "\n"
    if not os.path.exists(features_directory): #if path/subdirs do not exist then make them
        os.makedirs(features_directory)
    file_name = input_geojson_file.split('/')[-1].split('.')[0] #get filename without extensions
    path_X = features_directory+'/'+file_name+'_X_all' #new file path for X_all
    path_y = features_directory+'/'+file_name+'_y_all' # new file path for y_all
    if os.path.isfile(path_X): #check if X_all file already exists, then prompt for overwriting the file
        if yes:
            np.save(path_X, X_all)
        elif click.confirm(path_X.split('/')[-1] + " already exists, are you sure you want to overwrite?"):
            np.save(path_X, X_all)
    else:
        np.save(path_X, X_all)

    print ""

    if os.path.isfile(path_y): #check if y_all file already exists, then prompt for overwriting the file
        if yes:
            np.save(path_y, y_all)
        elif click.confirm(path_y.split('/')[-1] + " already exists, are you sure you want to overwrite?"):
            np.save(path_y, y_all)
    else:
        np.save(path_y, y_all)

    print "\ndone."



if __name__ == '__main__':
    main()