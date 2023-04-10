import json
import urllib.request
import os

wines = json.load(open("./vin_db.json"))

for i in range(len(wines)):
    try:
        os.makedirs(f'./data/{i}', exist_ok=True)
        urllib.request.urlretrieve(wines[i]['poster'], f'./data/{i}/image.png')
        text_file = open(f'./data/{i}/name.txt', 'w+')
        text_file.write(wines[i]['title'])
        text_file.close()
    except:
        print(f"ERROR WITH WINE NB {i} : {wines[i]['title']}")
