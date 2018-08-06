import gbdxtools
from gbdxtools import Interface, CatalogImage
import json
import rasterio
from rasterio.features import geometry_mask
from shapely import geometry
from shapely.geometry import shape
import copy
import geojson
import numpy as np
import numpy.ma as ma
import collections
from scipy import ndimage as ndi
from skimage import filters, color, exposure
import multiprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score
from pprint import pprint
from scipy.misc import imsave
from readjsonfromurl import fetch_jsons
from gridsearch import grid_search
import sys
#from PIL import Image
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
    """Remote sensing indices for vegetation, built-up, and bare soil."""

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
    # Normalize images for better comparison.
    image = (image - image.mean())/image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap') ** 2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap') ** 2)


def calc_gabors(image, frequency=1, theta_vals=[0, 1, 2, 3]):
    # convert to gray scale
    image = image.astype(np.uint8)
    rgb = np.dstack((image[2], image[3], image[5])) #B, G, R... correct?

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
    Returns image features of image bands, remote sensing indices, and gabor filters."""

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
    Keep burn scar polygons as red.
    Mark all building polygons labelled as ('yellow', False) and will be changed later."""

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
    if bbox == None:
        return CatalogImage(catalog_id, band_type="MS", pansharpen=pan)
    return CatalogImage(catalog_id, band_type="MS", pansharpen=pan, bbox=bbox)

def clean_data(X_all, y_all):
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

    X_all = X_all.astype(np.float32)
    X_all = X_all[0:].astype(np.float32)

    return X_all, y_all

def main():
    print "\nreading geojson..."
    # read jsons from remote
    js_list = fetch_jsons()

    #validate json for all geometries and correct labels:
    flag = -999
    ordinal = lambda n: "%d%s" % (n, "tsnrhtdd"[(n / 10 % 10 != 1) * (n % 10 < 4) * n % 10::4]) #from Gareth on codegolf: https://codegolf.stackexchange.com/questions/4707/outputting-ordinal-numbers-1st-2nd-3rd#answer-4712
    for j, js in enumerate(js_list):
        for i, feat in enumerate(js['features']):
            #print i
            if feat['properties']['Tomnod_label'] != True and feat['properties']['Tomnod_label'] != False and feat['properties']['Tomnod_label'] is not None:
                print >> sys.stderr, "Incorrect Tomnod_label at feature element {} in {} json".format(i, ordinal(j))
                flag = 1

            try:
                if feat['geometry']['type'] is not None and feat['geometry']['type'] != "Polygon":
                    print >> sys.stderr, "Incorrect feature type, must be Polygon only! Discovered at feature element {} in {} json".format(i, ordinal(j))
                    flag = 1
            except TypeError:
                print >> sys.stderr, "If this is of any concern, a NoneType for geometry type occured at feature element {} in {} json.".format(i, ordinal(j))
                continue


    #if flag is activated then do not execute rest
    if flag == 1:
        return

    print "\nloading CatalogImage..."
    image_list = [load_cat_image(js['properties']['catalog_id'], pan=True) for js in js_list]
    # TODO: remove bbox, read in catid from geojsons
    # image = load_cat_image('1040010038A0A900', (-122.711, 38.476, -122.686, 38.494), pan=True) #fountaingrove - big
    # print image.shape
    # print image.ipe.metadata['image']['acquisitionDate']
    # print image.ipe.metadata['image']['offNadirAngle']
    # print image.affine


    # TODO: validate json function here -> confirm all geometries are Polygons, expect properties labels "True/False/None"

    print "current number of features:", sum(len(js['features']) for js in js_list)

    print "\nremove Tomnod_labels = None..."
    filtered_feats_list = []
    for js in js_list:
        filtered_feats = filter(lambda x: x['properties']['Tomnod_label'] != None, js['features'])
        filtered_feats_list.append(filtered_feats)

    for i, js in enumerate(js_list):
        js['features'] = filtered_feats_list[i]
    print "new number of features:", sum(len(js['features']) for js in js_list)

    ## load the raster, mask it by the polygon
    print "\nmasking image..."
    polys = geojson_to_polygons(js_list[0])
    for i in range(1, len(js_list)):
        polys.extend(geojson_to_polygons(js_list[i]))
    image_aoi_segs, seg_masks = get_segment_masks(image, polys, invert=False) #toggle the inversion if necessary... remember this should be set to True for multiplying image to mask

    image_aoi_blobs = [ma.array(aoi, mask=np.dstack((seg_masks[i],)*8)) for i, aoi in enumerate(image_aoi_segs)] #image masks instead of just multiplying the image to mask
    #image_aoi_blobs = [np.multiply(aoi, seg_masks[i]) for i, aoi in enumerate(image_aoi_segs)]
    print len(image_aoi_blobs)
    print image_aoi_blobs[0].shape
    print type(image_aoi_blobs[0])

    # TODO: debug masked array in segments/pixels as features
    print "\ncompute segment prediction..."
    print "creating dataset"
    #creating segments as features
    p = multiprocessing.Pool(processes=4)
    rsi_samples = [p.apply_async(segment_as_feature, args=(blob,), kwds={'include_gabors':True}) for blob in image_aoi_blobs]
    rsi_samples_output = [p.get() for p in rsi_samples]
    print len(rsi_samples_output)

    #entire dataset
    X_all = np.vstack(rsi_samples_output)
    y_all = [x['properties']['Tomnod_label']==1 for x in js_list[0]['features']]
    for i in range(1, len(js_list)):
        y_all.extend([x['properties']['Tomnod_label']==1 for x in js_list[i]['features']])
    y_all = np.array(y_all)
    y_all.reshape(X_all.shape[0], 1)
    #y_all = np.array([x['properties']['Tomnod_label']==1 for x in js_copy['features']]).reshape(X_all.shape[0],1)

    #clean data
    # TODO: save out X_all, y_all -> check for html or local file path
    X_all, y_all = clean_data(X_all, y_all)

    # TODO: below this goes into train model script
    #stratified split sampling
    print "stratified split sampling"
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20, stratify=y_all, random_state=42)

    #create, train, and test model
    # classifier = RandomForestClassifier(n_estimators=50, max_features="sqrt", max_depth=None, min_samples_split=2, bootstrap=True, n_jobs=-1)
    # print "Parameters currently in use:"
    # pprint(classifier.get_params())
    # print "training model"
    # classifier.fit(X_train, y_train)
    # print classifier.score(X_train, y_train)
    # print "testing model"
    # predictions = classifier.predict(X_test)
    # trues = list(np.hstack(y_test))
    # print "\taccuracy =", accuracy_score(predictions, trues)
    # print "\tprecision =", precision_score(predictions, trues)
    # print "\trecall =", recall_score(predictions, trues)

    #save a image segment
    # blob0 = np.dstack((image_aoi_blobs[0][5], image_aoi_blobs[0][3], image_aoi_blobs[0][2]))
    # print blob0.shape
    #imsave('./blob0.png', blob0)

    better_model = grid_search(X_train, X_test, y_train, y_test)

    print "done."



if __name__ == '__main__':
    main()