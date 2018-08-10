"""""
    Author: Ai-Linh Alten
    Date created: 7/25/2018
    Date last modified: 7/25/2018
    Python Version: 2.7.15
"""

import json
import sys, getopt
import os
from pprint import pprint

def walk_dir(directory_path):
    files_in_dir = []
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            print "Appending:", filename
            files_in_dir.append(root+"/"+filename)

    return files_in_dir

def load_jsons(files):
    js_list = []
    for file in files:
        if file.find('.json') != -1:
            with open(file, 'r') as f:
                js = json.load(f)
                js_list.append(js)
        else:
            continue
    return js_list

def combine_jsons(js_list):
    #print js_list[0]['crs']
    new_js = {'type': js_list[0].get('type')}
    d_crs = {'crs': js_list[0].get('crs')}
    new_js.update(d_crs)
    d_props = {'properties': js_list[0].get('properties')}
    new_js.update(d_props)


    new_features = []
    for js in js_list:
         for feat in js['features']:
             new_features.append(feat)

    d_feats = {'features': new_features}
    new_js.update(d_feats)

    return new_js


if __name__ == "__main__":
    print "Combine_geojsons.py <path_to_json_tiles>"

    files_in_dir = walk_dir(sys.argv[1])

    print "loading jsons..."
    js_list = load_jsons(files_in_dir)

    print "combining jsons..."
    new_js = combine_jsons(js_list)
    print "new geojson created"

    #make directory for output of new geojson
    split_dirs = sys.argv[1].split('/')
    os.makedirs(sys.argv[1]+"/"+split_dirs[-1]+"_out")

    #new name for the geojson
    split_filenames = files_in_dir[0].split("/")
    names_in_file = split_filenames[-1].split("_")
    names_in_file.pop()
    new_filename = "_".join(names_in_file) + ".json"

    #write to geojson location
    with open(sys.argv[1]+"/"+split_dirs[-1]+"_out/"+new_filename, 'w') as f:
        json.dump(new_js, f)

    print "new geojson location:", sys.argv[1]+"/"+split_dirs[-1]+"_out/"+new_filename
