# DeepFake Detection

## Installation

1) Clone this repository.
```
https://github.com/Shahil98/Deep_Fake_Video_Detection.git
```

2) In the repository, execute `pip install -r requirements.txt` to install all the necessary libraries.

3) Download the pretrained weights.
	1) YOLO Face model [pretrained weights](https://drive.google.com/open?id=1PAdOJX1aMm-53bTdrCyUpFF_voZxQ4O_) and save it in /model_data/

## Usage

Use ``` python main.py --help   ``` to see usage for main.py.

```

usage: main.py [-h] [--genFaceCrops] [--frames [FRAMES]]
               [--videoDirectory [VIDEODIRECTORY]] [--trainEfficientNet]
               [--trainResNext] [--epochs [EPOCHS]] [--batchSize [BATCHSIZE]]
               [--imageDirectory [IMAGEDIRECTORY]]
               [--learningRate [LEARNINGRATE]]
               [--trainableLayers [TRAINABLELAYERS]] [--test]
               [--testVideoDirectory [TESTVIDEODIRECTORY]]

optional arguments:
  -h, --help            show this help message and exit
  --genFaceCrops        Will generate facecrops for videos inside
                        videoDirectory.
  --frames [FRAMES]     Will set the number of frames to be considered for
                        each video. Defaults to 30.
  --videoDirectory [VIDEODIRECTORY]
                        Will set the directory for videos whose facecrops are
                        to be generated. Defaults to 'test_videos'
  --trainEfficientNet   Will train EfficientNet.
  --trainResNext        Will train ResNeXt.
  --epochs [EPOCHS]     Will set the number of epochs for training. Defaults
                        to 200.
  --batchSize [BATCHSIZE]
                        Will set the batch size for training. Defaults to 64.
  --imageDirectory [IMAGEDIRECTORY]
                        Will set the directory for training set images.
                        Defaults to 'train_images'
  --learningRate [LEARNINGRATE]
                        Will set the learning rate for training. Defaults to
                        0.0001
  --trainableLayers [TRAINABLELAYERS]
                        Will set the number of trainable layers. Defaults to
                        5.
  --test                Test mode.
  --testVideoDirectory [TESTVIDEODIRECTORY]
                        Will set the directory for test videos. Defaults to
                        'test'

```
 
## Methodology

Implementation for DeepFake Detection Using Ensembling Techniques

![Alt Text](https://github.com/Shahil98/Deep_Fake_Video_Detection/blob/master/extra/arch.png)

**Determining whether a given video is Real or Fake by cropping the face of a person and classifying the
cropped image by an ensemble of 2 models (ResNext and EfficientNetB6)**
