import requests, zipfile, io, json

def get_json_remote(link):
    r = requests.get(link)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    print z.printdir()
    file = z.open('Fountaingrove.json')
    js = json.load(file)

    return js

def fetch_jsons():
    js_list = []
    remote_links = ['https://github.com/aalten77/WildfiresPilot/raw/master/data/Tubbs/Fountaingrove/Fountaingrove.json.zip', ]
