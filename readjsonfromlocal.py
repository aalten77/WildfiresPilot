import zipfile, json
import os
from os import walk

full_path = './data/Tubbs/Fountaingrove'
file_names = []

def get_local_file_names():
    return file_names

def get_json_local(path):
    if not os.path.exists(path):
        return -1
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
