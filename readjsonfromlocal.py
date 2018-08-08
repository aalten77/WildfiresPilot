import zipfile, json
from os import walk

full_path = './data/Tubbs/Fountaingrove'
file_names = []

def get_local_file_names():
    return file_names

def get_json_paths(path):
    z = zipfile.ZipFile(path, 'r')
    print z.printdir()
    file = z.open(path.split('/')[-1].replace('.zip',''))
    js = json.load(file)

    return js

def fetch_local_jsons():
    js_list = []

    for (root, dirs, files) in walk(full_path):
        for file in files:
            if file.find('.zip') != -1:
                file_names.append(file)
                js_list.append(get_json_paths(root+'/'+file))


    return js_list
