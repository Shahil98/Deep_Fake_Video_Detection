from keras import backend as K
from os.path import expanduser
import numpy as np
from keras.models import load_model
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from keras.layers import Input
from colorsys import hsv_to_rgb
from keras.utils import multi_gpu_model
from yolo3.utils import letterbox_image
from PIL import Image


class YOLO(object):
    """
    This class represents yolo and helps in loading the model
    and generating boxes around people's faces in an image.
    """
    _defaults = {
        "model_path": "model_data/yolo.h5",
        "anchors_path": "model_data/yolo_anchors.txt",
        "classes_path": "model_data/coco_classes.txt",
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        """
        This function returns a required default value given key.
        Args:
            n ('str'): Key value to be used to
                       retrieve from defaults dictionary.

        Returns:
            value: Returns corresponding value for the provided key.
        """
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def load_models(self, **kwargs):
        """
        This function is used to load the
        necessary things required for the yolo model.
        """
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        """
        This function is used to get the class
        names for which the yolo model will look for in the image.

        Returns:
            list[str]: A list of string representing different classes.
        """
        classes_path = expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        """
        This function is used to get the anchor dimensions.

        Returns:
            list[list[int]]: List having dimensions for each anchor.
        """
        anchors_path = expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(",")]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = expanduser(self.model_path)
        assert model_path.endswith(".h5"), (
                "Keras model or weights must be a .h5 file.")
        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except Exception as e:
            print("Exception occured ", e, "as normal model did not load")
            self.yolo_model = (
                tiny_yolo_body(
                    Input(shape=(None, None, 3)), num_anchors // 2, num_classes
                )
                if is_tiny_version
                else yolo_body(
                    Input(shape=(None, None, 3)), num_anchors // 3, num_classes
                )
            )
            self.yolo_model.load_weights(
                self.model_path
            )  # make sure model, anchors and classes match
        else:
            assert (self.yolo_model.layers[-1].output_shape[-1] ==
                    num_anchors / len(
                self.yolo_model.output
            ) * (
                num_classes + 5
            )), "Mismatch between model and given anchor and class sizes"

        hsv_tuples = [
            (x / len(self.class_names), 1.0, 1.0)
            for x in range(len(self.class_names))
        ]
        self.colors = list(map(lambda x: hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(
                lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors,
            )
        )
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(
            self.colors
        )  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = (multi_gpu_model(self.yolo_model,
                                               gpus=self.gpu_num))
        boxes, scores, classes = yolo_eval(
            self.yolo_model.output,
            self.anchors,
            len(self.class_names),
            self.input_image_shape,
            score_threshold=self.score,
            iou_threshold=self.iou,
        )
        return boxes, scores, classes

    def detect_image(self, image):
        """
        This function is used to get the boxes for each person in the images.

        Args:
            image (PIL.Image): A PIL Image object

        Returns:
            image (PIL.Image): A PIL Image object
            list[list[int]]: A list describing the
                             coordinates for each face in the image.
        """
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, (
                "Multiples of 32 required")
            assert self.model_image_size[1] % 32 == 0, (
                "Multiples of 32 required")
            boxed_image = letterbox_image(
                image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (
                image.width - (image.width % 32),
                image.height - (image.height % 32),
            )
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype="float32")
        image_data /= 255.0
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0,
            },
        )
        person_boxes = []
        for i, c in reversed(list(enumerate(out_classes))):
            box = out_boxes[i]
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype("int32"))
            left = max(0, np.floor(left + 0.5).astype("int32"))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype("int32"))
            right = min(image.size[0], np.floor(right + 0.5).astype("int32"))
            height = bottom - top
            width = right - left
            top = top - int(height * 0.1)
            bottom = bottom + int(height * 0.1)
            left = left - int(width * 0.1)
            right = right + int(width * 0.1)
            top = max(0, top)
            left = max(0, left)
            right = min(image.size[0], right)
            bottom = min(image.size[1], bottom)
            coordinates = [top, bottom, left, right]
            person_boxes += [coordinates]
        return image, person_boxes

    def close_session(self):
        """
        This function is used to close the session.
        """
        self.sess.close()

    def detect_video(self, frame):
        """
        This function is used to get face bounding boxes for a frame.

        Args:
            frame: Image read using opencv.

        Returns:
            list[list[int]]: A list describing the
                             coordinates for each face in the image.
        """
        image = Image.fromarray(frame)
        image, person_boxes = self.detect_image(image)
        return person_boxes
