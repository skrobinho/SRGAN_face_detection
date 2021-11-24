import face_recognition
import os
import cv2
import pickle
import time
import srgan
import data_loader
import numpy as np


def main():

    # Defining constants
    KNOWN_FACES_DIR = "known_faces"
    TOLERANCE = 0.55
    FRAME_THICKNESS = 3
    FONT_SCALE = 0.5
    MODEL = "cnn"  # "hog"

    # Loading GAN weights
    gan = srgan.SRGAN(training=False)
    gan.load_weights('./weights/4X_generator.h5')

    # Loading video
    video = cv2.VideoCapture("./test_videos/cabinet_meeting_2.mp4")

    print("loading known faces")

    known_faces = []
    known_names = []

    if not os.path.exists(KNOWN_FACES_DIR):
        os.mkdir(KNOWN_FACES_DIR)

    # Loading known faces encodings
    for name in os.listdir(KNOWN_FACES_DIR):
        for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = face_recognition.load_image_file(
                    f"{KNOWN_FACES_DIR}/{name}/{filename}")
                encoding = face_recognition.face_encodings(image)[0]
            elif filename.lower().endswith('.pkl'):
                with open(f'{KNOWN_FACES_DIR}/{name}/{filename}', 'rb') as pickle_file:
                    encoding = pickle.load(pickle_file)
            known_faces.append(encoding)
            try:
                known_names.append(int(name))
            except ValueError:
                known_names.append(name)

    # Indexing faces
    known_names_int = [i for i in known_names if type(i) == int]
    if len(known_names_int) > 0:
        next_id = max(known_names_int) + 1
    else:
        next_id = 0

    print("processing unknown faces")

    # Processing unknown faces
    while True:
        _, image = video.read()

        # Saving locations and encodings of detected faces
        locations = face_recognition.face_locations(image, model=MODEL)
        encodings = face_recognition.face_encodings(image, locations)

        for face_encoding, face_location in zip(encodings, locations):
            results = face_recognition.compare_faces(
                known_faces, face_encoding, TOLERANCE)
            match = None

        # Checking if detected face is being recognized as one of known faces
            if True in results:
                match = str(known_names[results.index(True)])
                print(f"Match found: {match}")
            else:
                match = str(next_id)
                next_id += 1
                known_names.append(match)
                known_faces.append(face_encoding)
                os.mkdir(f"{KNOWN_FACES_DIR}/{match}")
                pickle.dump(face_encoding, open(
                    f"{KNOWN_FACES_DIR}/{match}/{match}-{int(time.time())}.pkl", "wb"))

            dirname = 'faces_pics'
            if not os.path.exists(f'{dirname}/{match}'):
                os.makedirs(f'{dirname}/{match}')

            roi_color = image[face_location[0]:face_location[2],
                            face_location[3]:face_location[1]]
            cv2.imwrite(f'{dirname}/{match}/{int(time.time())}.jpg', roi_color)

            height = abs(face_location[2] - face_location[0])
            width = abs(face_location[1] - face_location[3])

            # Using SRGAN to improve resolution
            img_lr = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
            img_lr = data_loader.DataLoader.resize_img(
                img_lr, size=(width, height))
            img_lr = img_lr / 127.5 - 1

            img_sr = np.squeeze(
                gan.generator.predict(
                    np.expand_dims(img_lr, 0),
                    batch_size=1
                ),
                axis=0
            )

            img_sr = ((img_sr+1)*127.5).astype(np.uint8)
            img_lr = ((img_lr+1)*127.5).astype(np.uint8)

            cv2.imwrite(f'{dirname}/{match}/{int(time.time())}_lr.jpg',
                        cv2.cvtColor(img_lr, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f'{dirname}/{match}/{int(time.time())}_sr.jpg',
                        cv2.cvtColor(img_sr, cv2.COLOR_RGB2BGR))

            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            color = [0, 255, 0]
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2]+22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image, match, (face_location[3]+10, face_location[2]+15),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (200, 200, 200))

        cv2.imshow("", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

if __name__ == '__main__':
    main()