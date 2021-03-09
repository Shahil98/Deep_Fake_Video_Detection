from os import getcwd
from os.path import isdir


def test_face_crops():
    face_crops_dir = getcwd()+"/code/FaceCrops"
    assert isdir(face_crops_dir) == 1

    face_crops_person_dir = getcwd()+"/code/FaceCrops/aaqaifqrwn/person1"
    assert isdir(face_crops_person_dir) == 1
