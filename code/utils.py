import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from keras.preprocessing import image


class Utils:
    def __init__(self, frames_per_video):
        self.frames_per_video = frames_per_video
        self.input_size = 150
        self.input_size2 = 224

    """
    Function :- isotropically_resize_image
    Resize the face image as per the input size of the inception model.
    Input :
           img :- Face Crops generated from the main frame of the video.
           size :- Width/Height of the image.
                   (Image is supposed to be a square)
    Returns :
            resized :- Resized image with it's higher dimension equal to size.
    """

    def isotropically_resize_image(self, img, size, resample=cv2.INTER_AREA):
        h, w = img.shape[:2]
        if w > h:
            h = h * size // w
            w = size
        else:
            w = w * size // h
            h = size

        resized = cv2.resize(img, (w, h), interpolation=resample)
        return resized

    """
    Function :- make_square_image
    Add borders to the image in order to change the shape of image to a square.
    Input :
           img :- Resized face crops from the
                  isotropically_resize_image function.
    Returns :
            square_image :- Square image with a border
                            of height (size-h) and width (size-w).
    """

    def make_square_image(self, img):
        h, w = img.shape[:2]
        size = max(h, w)
        b = size - h
        r = size - w
        return cv2.copyMakeBorder(
            img, 0, b, 0, r, cv2.BORDER_CONSTANT, value=0)

    """
    Function :- find_number_of_person
    Return the number of person id found after
    completion of detection and tracking algotihms.
    Input :
           video_face_path :- Path of the folder of a specific
           video where different person ID are stored.
    Returns :
            arr :- Number of different people in the entire video.
    """

    def find_number_of_person(self, video_face_path):
        arr = listdir(video_face_path)
        return len(arr)

    def generate_image_tensor(self, video_face_path):
        person_count = self.find_number_of_person(video_face_path)
        person_x = []
        for i in range(1, person_count + 1):
            mypath = video_face_path + "/Person" + str(i)
            onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
            x = np.zeros(
                (self.frames_per_video, self.input_size, self.input_size, 3),
                dtype=np.uint8,
            )
            n = 0
            for f in onlyfiles:
                files = mypath + "/" + f
                img = cv2.imread(files)
                frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                resized_face = self.isotropically_resize_image(frame,
                                                               self.input_size)
                resized_face = self.make_square_image(resized_face)
                x[n] = resized_face
                n += 1
            person_x.append(x)
        return person_x

    def generate_image_tensor2(self, video_face_path):
        person_count = self.find_number_of_person(video_face_path)
        person_x = []
        for i in range(1, person_count + 1):
            mypath = video_face_path + "/Person" + str(i)
            onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
            x = np.zeros(
                (self.frames_per_video, self.input_size2, self.input_size2, 3),
                dtype=np.uint8,
            )
            n = 0
            for f in onlyfiles:
                files = mypath + "/" + f
                img = image.load_img(files, target_size=(224, 224))
                img_tensor = image.img_to_array(img)
                img_tensor /= 255.0
                x[n] = img_tensor
                n += 1
            person_x.append(x)
        return person_x
