import os
import json

folder = r'/data/dataset/dev/aac'
path_spkid = []
for i,subfolder in enumerate(os.listdir(folder)):
    for sf in os.listdir(os.path.join(folder, subfolder)):
        for file in os.listdir(os.path.join(folder, subfolder, sf)):
            wavfile = os.path.join(folder, subfolder, sf, file)
            path_spkid.append({"path": wavfile, "spkid": i})

file = open('vox.json', 'w')
json.dump(path_spkid, file)
