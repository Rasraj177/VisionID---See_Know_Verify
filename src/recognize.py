import cv2
import face_recognition
import pickle
import os


known_encodings = []
known_names = []


encoding_folder = "encodings"


for filename in os.listdir(encoding_folder):

    if filename.endswith(".pkl"):

        file_path = os.path.join(
            encoding_folder,
            filename
        )

        with open(file_path, "rb") as file:
            encoding = pickle.load(file)

        known_encodings.append(encoding)

        name = os.path.splitext(filename)[0]

        known_names.append(name)


print("Loaded registered faces:")
print(known_names)


cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    face_locations = face_recognition.face_locations(
        rgb_frame
    )

    face_encodings = face_recognition.face_encodings(
        rgb_frame,
        face_locations
    )


    for face_encoding, location in zip(
        face_encodings,
        face_locations
    ):

        name = "Unknown"

        if known_encodings:

            matches = face_recognition.compare_faces(
                known_encodings,
                face_encoding
            )

            face_distances = face_recognition.face_distance(
                known_encodings,
                face_encoding
            )

            best_match_index = face_distances.argmin()

            if matches[best_match_index]:
                name = known_names[best_match_index]


        top, right, bottom, left = location


        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            name,
            (left, bottom + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    cv2.imshow(
        "VisionID - Face Recognition",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()