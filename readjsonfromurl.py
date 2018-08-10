import requests, zipfile, io, json

remote_links = ['https://github.com/aalten77/WildfiresPilot/raw/master/data/Tubbs/Fountaingrove/Fountaingrove.json.zip',
                    'https://github.com/aalten77/WildfiresPilot/raw/master/data/Tubbs/Fountaingrove/Fountaingrove_v2.json.zip',
                    'https://github.com/aalten77/WildfiresPilot/raw/master/data/Tubbs/Fountaingrove/Fountaingrove_v3.json.zip']#,
                    #'https://github.com/aalten77/WildfiresPilot/raw/master/data/Tubbs/Fountaingrove/fake.json.zip']

def get_remote_links():
    return remote_links

def get_json_remote(link):
    r = requests.get(link)
    if r.status_code == 404:
        return -1
    if link.find('.zip') != -1:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        print z.printdir()
        file = z.open(link.split('/')[-1].replace('.zip',''))
        js = json.load(file)
    else:
        js = json.loads(r.content)

    return js

def fetch_jsons():
    js_list = []

    for link in remote_links:
        js_list.append(get_json_remote(link))

    return js_list
