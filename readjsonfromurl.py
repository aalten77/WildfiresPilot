import requests, zipfile, io, json

r = requests.get('https://github.com/aalten77/WildfiresPilot/raw/master/data/Tubbs/Fountaingrove/Fountaingrove.json.zip')
z = zipfile.ZipFile(io.BytesIO(r.content))
print z.printdir()
file = z.open('Fountaingrove.json')
js = json.load(file)

print len(js['features'])