# ghostvlad-speaker
An tensorflow implementation of ghostvlad, the keras version [VGG-Speaker-Recognition](https://github.com/WeidiXie/VGG-Speaker-Recognition)

## Steps:
#### 1. generate speaker labels from dataset [voxceleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
`python voxcele2json.py`

#### 2. Training
`python train.py`</br>
Please change the `args_params` in __main__

#### 3. predict
`python predict.py`

