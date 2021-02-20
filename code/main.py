import numpy as np
from os import listdir, mkdir
from os.path import isdir
from keras.models import Model
from keras.preprocessing import image
from keras import (
    optimizers,
    backend as keras_backend,
    layers as keras_layers,
    models as keras_models,
    utils as keras_utils,
)
from keras_applications.resnext import ResNeXt101
import time
from sklearn import metrics
from efficientnet.keras import EfficientNetB6
import json
from faceCrops import GenerateFaceCrops
import argparse


class Result:
    """
    Function :- generate_report
    This function is called in the after the predicted
    scores of every video in the directory has been
    generated. It is mainly used to generate confusion
    matrix, classification report and accuracy.
    Input :
            probability_scores :- Dictionary with video names as their
                                  keys and there predicted probability score
                                  as the value of that specific key.
            threshold :- Real value in between 0 and 1.
                         If probability is more than threshold,
                         then we will consider video as fake.
    Returns :
            probability_scores :- Dictionary with video names as their
            keys and there predicted probability score as the
            value of that specific key.
    """

    def generate_report(self, probability_scores, threshold):
        y_real = []
        y_pred = []
        y_prob = []
        for keys in probability_scores.keys():
            label = int(keys.split("_")[-1])
            y_real.append(label)
            if probability_scores[keys] > threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
            y_prob.append(probability_scores[keys])

        print(metrics.confusion_matrix(y_real, y_pred))
        print(metrics.accuracy_score(y_real, y_pred))
        print(metrics.classification_report(y_real, y_pred))
        print("Log Loss value is :-", metrics.log_loss(y_real, y_prob))

    def ensemble(self, scores_resnext, scores_efficient):
        ensembled = {}
        for keys in scores_resnext.keys():
            ensembled[keys] = (0.5 * scores_efficient[keys]
                               + 0.5 * scores_resnext[keys])
        return ensembled

    def dump_json(self, file_name, dictionary):
        with open(file_name + ".json", "w") as fp:
            json.dump(dictionary, fp)

    def read_json(self, file_name):
        with open(file_name + ".json") as f:
            scores1 = json.load(f)
        return scores1


class ResNext:
    def __load_model(self, learning_rate, trainable_layers):
        print("MODEL LOADING..........")
        in_lay = keras_layers.Input(shape=(224, 224, 3))
        base_model = model = ResNeXt101(
            input_tensor=keras_layers.Input(shape=(224, 224, 3)),
            include_top=False,
            weights="imagenet",
            backend=keras_backend,
            layers=keras_layers,
            models=keras_models,
            utils=keras_utils,
        )

        base_layer = base_model(in_lay)
        pooling_layer = keras_layers.GlobalAveragePooling2D()(base_layer)
        linear_layer = keras_layers.Dense(2048,
                                          activation="relu")(pooling_layer)
        out_layer = keras_layers.Dense(1, activation="sigmoid")(linear_layer)
        model = Model(inputs=[in_lay], outputs=[out_layer])
        model.compile(
            loss="binary_crossentropy",
            optimizer=optimizers.Adam(lr=learning_rate),
            metrics=["accuracy"],
        )

        for layer_index in range(len(model.layers) - trainable_layers):
            model.layers[layer_index].trainable = False

        return model

    def train(self,
              train_set_dir,
              epochs,
              batch_size,
              learning_rate,
              trainable_layers):

        train_datagen = image.ImageDataGenerator(rescale=1.0 / 255)
        train_generator = train_datagen.flow_from_directory(
            train_set_dir,
            target_size=(224, 224),
            class_mode="binary",
            color_mode="rgb",
            batch_size=batch_size,
        )
        model = self.__load_model(learning_rate, trainable_layers)

        model.fit_generator(train_generator, epochs=epochs)

        if not isdir("./Weights"):
            mkdir("Weights")

        model.save_weights("Weights/ResnextDeepFakes.h5")

    def predict(self, test_set_dir):
        model = self.load_efficient_attention()
        print("Model has been loaded")
        model.load_weights("Weights/ResnextDeepFakes.h5")
        test_datagen = image.ImageDataGenerator(rescale=1.0 / 255)

        arr = listdir(test_set_dir)
        probability_scores = {}

        for video in arr:
            start = time.time()
            val_set_dir = test_set_dir + "/" + video
            person_count = len(listdir(val_set_dir))

            test_generator = test_datagen.flow_from_directory(
                val_set_dir,
                target_size=(224, 224),
                batch_size=10,
                class_mode="binary",
                color_mode="rgb",
                shuffle=True,
            )
            test_generator.reset()
            predictions = model.predict_generator(
                test_generator, steps=test_generator.samples // 10
            )
            per_person = len(predictions) // person_count
            y_person = []
            for i in range(person_count):
                y_person.append(
                    predictions[i * per_person: i * per_person + per_person]
                    .mean()
                    .item()
                )
            probability_scores[video] = max(y_person)
            end = time.time()
            print(video, ":-",
                  probability_scores[video],
                  "Time Taken :-",
                  end - start)

        return probability_scores


class EfficientNetWithAttention:
    def __load_model(self, learning_rate, trainable_layers):
        print("MODEL LOADING..........")
        in_lay = keras_layers.Input(shape=(224, 224, 3))
        base_model = EfficientNetB6(
            input_shape=(224, 224, 3), weights="imagenet", include_top=False
        )

        pt_features = base_model(in_lay)
        pt_depth = base_model.get_output_shape_at(0)[-1]
        bn_features = keras_layers.BatchNormalization()(pt_features)
        # here we do an attention mechanism to turn pixels in the GAP on an off
        attn_layer = keras_layers.Conv2D(
            64, kernel_size=(1, 1), padding="same", activation="relu"
        )(keras_layers.Dropout(0.5)(bn_features))
        attn_layer = keras_layers.Conv2D(
            16, kernel_size=(1, 1), padding="same", activation="relu"
        )(attn_layer)
        attn_layer = keras_layers.Conv2D(
            8, kernel_size=(1, 1), padding="same", activation="relu"
        )(attn_layer)
        attn_layer = keras_layers.Conv2D(
            1, kernel_size=(1, 1), padding="valid", activation="sigmoid"
        )(attn_layer)
        # fan it out to all of the channels
        up_c2_w = np.ones((1, 1, 1, pt_depth))
        up_c2 = keras_layers.Conv2D(
            pt_depth,
            kernel_size=(1, 1),
            padding="same",
            activation="linear",
            use_bias=False,
            weights=[up_c2_w],
        )
        up_c2.trainable = False
        attn_layer = up_c2(attn_layer)

        mask_features = keras_layers.multiply([attn_layer, bn_features])
        gap_features = keras_layers.GlobalAveragePooling2D()(mask_features)
        gap_mask = keras_layers.GlobalAveragePooling2D()(attn_layer)
        # to account for missing values from the attention model
        gap = keras_layers.Lambda(lambda x: x[0] / x[1], name="RescaleGAP")(
            [gap_features, gap_mask]
        )
        gap_dr = keras_layers.Dropout(0.25)(gap)
        dr_steps = keras_layers.Dropout(0.25)(
            keras_layers.Dense(128, activation="relu")(gap_dr)
        )
        out_layer = keras_layers.Dense(1, activation="sigmoid")(dr_steps)
        model = Model(inputs=[in_lay], outputs=[out_layer])
        model.compile(
            loss="binary_crossentropy",
            optimizer=optimizers.Adam(lr=learning_rate),
            metrics=["accuracy"],
        )

        for layer_index in range(len(model.layers) - trainable_layers):
            model.layers[layer_index].trainable = False

        return model

    def train(self,
              train_set_dir,
              epochs,
              batch_size,
              learning_rate,
              trainable_layers):

        train_datagen = image.ImageDataGenerator(rescale=1.0 / 255)
        train_generator = train_datagen.flow_from_directory(
            train_set_dir,
            target_size=(224, 224),
            class_mode="binary",
            color_mode="rgb",
            batch_size=batch_size,
        )
        model = self.__load_model(learning_rate, trainable_layers)

        model.fit_generator(train_generator, epochs=epochs)

        if not isdir("./Weights"):
            mkdir("Weights")

        model.save_weights("Weights/EFB6withAttentionDeepFakes.h5")

    def predict(self, test_set_dir):
        model = self.load_efficient_attention()
        print("Model has been loaded")
        model.load_weights("Weights/EFB6withAttentionDeepFakes.h5")
        test_datagen = image.ImageDataGenerator(rescale=1.0 / 255)

        arr = listdir(test_set_dir)
        probability_scores = {}

        for video in arr:
            start = time.time()
            val_set_dir = test_set_dir + "/" + video
            person_count = len(listdir(val_set_dir))

            test_generator = test_datagen.flow_from_directory(
                val_set_dir,
                target_size=(224, 224),
                batch_size=10,
                class_mode="binary",
                color_mode="rgb",
                shuffle=True,
            )
            test_generator.reset()
            predictions = model.predict_generator(
                test_generator, steps=test_generator.samples // 10
            )
            per_person = len(predictions) // person_count
            y_person = []
            for i in range(person_count):
                y_person.append(
                    predictions[i * per_person: i * per_person + per_person]
                    .mean()
                    .item()
                )
            probability_scores[video] = max(y_person)
            end = time.time()
            print(video, ":-",
                  probability_scores[video],
                  "Time Taken :-",
                  end - start)

        return probability_scores


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--genFaceCrops", action="store_true")
    parser.add_argument("--frames", nargs="?", default=30, type=int)
    parser.add_argument("--videoDirectory", nargs="?", default="test_videos")

    parser.add_argument("--trainEfficientNet", action="store_true")
    parser.add_argument("--trainResNext", action="store_true")
    parser.add_argument("--epochs", nargs="?", default=200, type=int)
    parser.add_argument("--batchSize", nargs="?", default=64, type=int)
    parser.add_argument("--imageDirectory", nargs="?", default="train_images")
    parser.add_argument("--learningRate",
                        nargs="?",
                        default=0.0001,
                        type=float)
    parser.add_argument("--trainableLayers", nargs="?", default=5, type=int)

    parser.add_argument("--test", action="store_true")
    parser.add_argument("--testVideoDirectory",
                        nargs="?",
                        default="test",
                        type=str)

    args = parser.parse_args()

    if args.genFaceCrops:
        face_crop_object = GenerateFaceCrops(args.videoDirectory, args.frames)
        face_crop_object.face_crops()

    elif args.trainResNext:
        resnext = ResNext(
            args.imageDirectory,
            args.epochs,
            args.batchSize,
            args.learningRate,
            args.trainableLayers,
        )
        resnext.train()
    elif args.trainEfficientNet:
        efficientnet_model_object = EfficientNetWithAttention()
        efficientnet_model_object.train(
            args.imageDirectory,
            args.epochs,
            args.batchSize,
            args.learningRate,
            args.trainableLayers,
        )
    elif args.test:
        resnext_object = ResNext()
        resnext_output = resnext_object.predict()

        efficientnet_object = EfficientNetWithAttention(
                                                    args.testVideoDirectory)
        efficientnet_output = efficientnet_object.predict(
                                                    args.testVideoDirectory)

        result_object = Result()
        ensemble_scores = result_object.ensemble(resnext_output,
                                                 efficientnet_output)
        result_object.generate_report(ensemble_scores, 0.5)

    else:
        print("Invalid arguments provided")
