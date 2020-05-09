# DeepFake Detection

Implementation for DeepFake Detection Using Ensembling Techniques

![Alt Text](https://github.com/kenil-shah/DeepFake_Detection/blob/master/extra/arch.png)

**Determining whether a given video is Real or Fake by cropping the face of a person and classifying the
cropped image by ensembling ResNext and EfficientNetB6**

## Installation

1) Clone this repository.
```
https://github.com/Shahil98/Deep_Fake_Video_Detection.git
```

2) In the repository, execute `pip install -r requirements.txt` to install all the necessary libraries.

3) Three deep learning models are used inorder to determine the class of video
	1) *YOLO Face model:- Used to determine the coordinates of the face of person and generate a cropped facial image using those coordinates*
	2) *ResNext:- First Model used for ensembling*
	3) *EfficientNetB6(With Attention):- Second Model used for ensembling*

4) Download the pretrained weights.
	1) YOLO Face model [pretrained weights](https://drive.google.com/open?id=1PAdOJX1aMm-53bTdrCyUpFF_voZxQ4O_/view?usp=sharing) and save it in /model_data/
	2) ResNext:- [pretrained weights](https://drive.google.com/open?id=1PC0PAQNTbDVIOBkZ_qkCTpEj9lgVK-jI/view?usp=sharing) and save it in the root directory 
	3) EfficientNetB6(With Attention)[pretrained weight](https://drive.google.com/open?id=1fwGpRb5oWM8zjzxyBljLv7cVPczUhEzh) and save it in the root directory
