import gbdxtools
from gbdxtools import Interface, CatalogImage
import json
import rasterio
from rasterio.features import geometry_mask
#from rasterio.mask import mask
from shapely import geometry
from shapely.geometry import shape
import copy
import geojson
import numpy as np
from scipy.misc import imsave
#from PIL import Image
gbdx = Interface()

def geojson_to_polygons(js_):
    """Convert the geojson into Shapely Polygons.
    Keep burn scar polygons as red.
    Mark all building polygons labelled as ('yellow', False) and will be changed later."""

    polys = []
    #burnt_polys = []
    #building_polys = []
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


        # if feat['properties']['color'] == 'red':  # red for the burnt region
        #     burnt_polys.append(g2)
        # else:  # for the building poly
        #     building_polys.append([g2, [feat['properties']['BuildingID'], 'yellow',
        #                                 False]])  # mark building polygons as 'yellow' for non-burnt for now
    #return burnt_polys, building_polys

def get_segment_masks(image, polys):
    image_aoi_segs = []
    seg_masks = []

    for poly in polys:
        bounds = poly.bounds
        image_aoi_seg = image.aoi(bbox=list(bounds))
        segment_mask = geometry_mask([poly], transform=image_aoi_seg.affine,
                                 out_shape=(image_aoi_seg.shape[1], image_aoi_seg.shape[2]), invert=True)

        image_aoi_segs.append(image_aoi_seg)
        seg_masks.append(segment_mask)

    return image_aoi_segs, seg_masks

def load_cat_image(catalog_id, bbox, pan=False):
    return CatalogImage(catalog_id, band_type="MS", pansharpen=pan, bbox=bbox)

def main():
    print "\nloading CatalogImage..."
    image = load_cat_image('1040010038A0A900', (-122.711, 38.476, -122.686, 38.494), pan=True) #fountaingrove - big
    print image.shape
    print image.ipe.metadata['image']['acquisitionDate']
    print image.ipe.metadata['image']['offNadirAngle']
    print image.affine

    print "\nreading geojson..."
    with open('data/Tubbs/Fountaingrove/Fountaingrove.json') as data_file:
        js = json.load(data_file)

    js_copy = copy.deepcopy(js)
    print "current number of features:", len(js['features'])
    print js['features'][0]['geometry'].keys()
    print type(js['features'][0].get('geometry'))
    print js['features'][0]['properties'].keys()
    # for i, feat in enumerate(js['features']):
    #     #     if feat['properties']['Tomnod_label'] == None:

    print "\nremove Tomnod_labels = None..."
    filtered_feats = filter(lambda x: x['properties']['Tomnod_label'] != None, js_copy['features'])

    #print filtered_feats[0]['properties']['Tomnod_label']
    js_copy['features'] = filtered_feats
    print "new number of features:", len(js_copy['features'])

    ## load the raster, mask it by the polygon
    print "\nmasking image..."
    polys = geojson_to_polygons(js_copy)
    image_aoi_segs, seg_masks = get_segment_masks(image, polys)

    image_aoi_blobs = [np.multiply(aoi, seg_masks[i]) for i, aoi in enumerate(image_aoi_segs)]
    print len(image_aoi_blobs)
    print image_aoi_blobs[0].shape
    print type(image_aoi_blobs[0])

    blob0 = np.dstack((image_aoi_blobs[0][5], image_aoi_blobs[0][3], image_aoi_blobs[0][2]))
    print blob0.shape
    imsave('./blob0.png', blob0)
    #im = Image.fromarray(blob0)
    #im.save('./blob0.jpeg')
    #tif = image_aoi_blobs[0].geotiff(path='./blob0.tif', proj='EPSG:4326', bands=[4,2,1])
    print "done."
    #out = geometry_mask([js_copy['features'][0]['geometry']], out_shape=)


    # with rasterio.open('./output.tif') as src:
    #     out_image, out_transform = mask(src, [js_copy['features'][0]['geometry']], crop=True)
    # out_meta = src.meta.copy()
    #
    # print "out image shape:", out_image.shape
    #
    # out_meta.update({'driver': 'GTiff',
    #                  'height': out_image.shape[1],
    #                  'width': out_image.shape[2],
    #                  'transform': out_transform})
    #
    # with rasterio.open('./mask_0_nodata.tif', 'w', **out_meta) as dest:
    #     dest.write(out_image)



if __name__ == '__main__':
    main()