import requests, zipfile, io, json

def get_json_remote(link):
    r = requests.get(link)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    print z.printdir()
    file = z.open(link.split('/')[-1].replace('.zip',''))
    js = json.load(file)

    return js

def fetch_jsons():
    js_list = []
    remote_links = ['https://github.com/aalten77/WildfiresPilot/raw/master/data/Tubbs/Fountaingrove/Fountaingrove.json.zip',
                    'https://github.com/aalten77/WildfiresPilot/raw/master/data/Tubbs/Fountaingrove/Fountaingrove_v2.json.zip',
                    'https://github.com/aalten77/WildfiresPilot/raw/master/data/Tubbs/Fountaingrove/Fountaingrove_v3.json.zip']


    for link in remote_links:
        js_list.append(get_json_remote(link))

    return js_list
