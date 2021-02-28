from os import mkdir, listdir
from os.path import isfile, join, isdir
from yolo import YOLO
import cv2


def face_crops(video_directory,
               sample_rate,
               directory="FaceCrops"):
    """
    This function is used to generate face crops of people from videos.

    Args:
        video_directory (str): Path to the directory for videos.
        sample_rate (int): Number of frames to be considered
                           for getting face crops in a video.
        directory (str): Name of the folder where face
                         crops will be stored. Defaults to "FaceCrops".
    """
    try:
        mkdir(directory)
    except Exception as e:
        print(e)
    videos = [
        f
        for f in listdir(video_directory)
        if isfile(join(video_directory, f))
    ]
    yolo_object = YOLO()
    yolo_object.load_models()
    for v in videos:
        try:
            video_path = video_directory + "/" + v
            video_name = video_path.split("/")[-1].split(".")[0]
            if not isdir("./"+directory + "/" + video_name):
                mkdir(directory + "/" + video_name)
            vid = cv2.VideoCapture(video_path)
            ret, frame = vid.read()
            person_boxes = yolo_object.detect_video(frame)
            for i in range(0, len(person_boxes)):
                if not isdir("./"
                             + directory
                             + "/"
                             + video_name
                             + "/"
                             + "Person"
                             + str(i + 1)):
                    mkdir(directory
                          + "/"
                          + video_name
                          + "/"
                          + "Person"
                          + str(i + 1))
            person_no = 1
            for bbox in person_boxes:
                x, y, w, h = (
                    int(bbox[2]),
                    int(bbox[0]),
                    int(bbox[3] - bbox[2]),
                    int(bbox[1] - bbox[0]),
                )
                img = frame[y: y + h, x: x + w]
                cv2.imwrite(
                    directory
                    + "/"
                    + video_name
                    + "/"
                    + "Person"
                    + str(person_no)
                    + "/"
                    + "Frame0.jpg",
                    img,
                )
                person_no += 1
            multiTracker = cv2.MultiTracker_create()
            for i in range(0, len(person_boxes)):
                bbox = person_boxes[i]
                x, y, w, h = (
                    int(bbox[2]),
                    int(bbox[0]),
                    int(bbox[3] - bbox[2]),
                    int(bbox[1] - bbox[0]),
                )
                bbox = [x, y, w, h]
                multiTracker.add(cv2.TrackerKCF_create(),
                                 frame, tuple(bbox))
            frame_number = 1
            total_frames = 1
            while vid.isOpened():
                skip_rate = (vid.get(cv2.CAP_PROP_FRAME_COUNT) //
                             sample_rate)
                ret, frame = vid.read()
                if not ret:
                    break
                ret, boxes = multiTracker.update(frame)
                if total_frames == sample_rate:
                    break
                if frame_number % skip_rate != 0:
                    frame_number += 1
                    continue
                total_frames += 1
                person_no = 1
                for i, newbox in enumerate(boxes):
                    x, y, w, h = newbox
                    img = frame[int(y): int(y + h), int(x): int(x + w)]
                    cv2.imwrite(
                        directory
                        + "/"
                        + video_name
                        + "/"
                        + "Person"
                        + str(person_no)
                        + "/"
                        + "Frame"
                        + str(frame_number)
                        + ".jpg",
                        img,
                    )
                    person_no += 1
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                frame_number += 1
        except Exception as e:
            print(e)
            pass
