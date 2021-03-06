"""""
    Author: Ai-Linh Alten
    Date created: 8/7/2018
    Date last modified: 8/10/2018
    Python Version: 2.7.15
"""

import zipfile, json
import os
from os import walk

full_path = './data/Tubbs/Fountaingrove'
file_names = []

def get_local_file_names():
    return file_names

def get_json_local(path):

    """
    Fetch json from local path.
    :param path: path to json from local file system. Can be .json or .json.zip extension.
    :return: return loaded json
    """
    #if the path doesn't exist then exit
    if not os.path.exists(path):
        return -1

    #If path is a zip, then unzip the json. Else, load json.
    if path.find('.zip') != -1:
        z = zipfile.ZipFile(path, 'r')
        print z.printdir()
        file = z.open(path.split('/')[-1].replace('.zip',''))
        js = json.load(file)
    else:
        with open(path) as f:
            js = json.load(f)

    return js

def fetch_local_jsons():
    js_list = []

    for (root, dirs, files) in walk(full_path):
        for file in files:
            if file.find('.zip') != -1:
                file_names.append(file)
                js_list.append(get_json_local(root+'/'+file))


    return js_list
