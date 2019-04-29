# ghostvlad-speaker
An tensorflow implementation of ghostvlad, pretrained model can be downloaded in [ghostvlad-speaker.tar](https://drive.google.com/open?id=1VusSJhDJCrtavsGwMQTUP30HxyFoDPRh)

## prerequisit
```
1. pydub, also need ffmpeg to be installed.
2. tensorflow
3. librosa
```

## Steps:
#### 1. generate speaker labels from dataset [voxceleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
`python voxcele2json.py`

#### 2. Training
`python train.py`</br>
Please change the `args_params` in __main__

#### 3. predict
`python predict.py`

