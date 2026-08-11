import cv2
import face_recognition
import os


KNOWN_FACES_DIR = "known_faces"

known_face_encodings = []
known_face_names = []


# ---------------------------------------
# LOAD REGISTERED FACES
# ---------------------------------------

for filename in os.listdir(KNOWN_FACES_DIR):

    if not filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        continue

    file_path = os.path.join(
        KNOWN_FACES_DIR,
        filename
    )

    image = face_recognition.load_image_file(
        file_path
    )

    encodings = face_recognition.face_encodings(
        image
    )

    if len(encodings) == 0:
        print("No face found in:", filename)
        continue

    encoding = encodings[0]

    known_face_encodings.append(encoding)

    name = os.path.splitext(filename)[0]

    known_face_names.append(name)

    print("Loaded:", name)


# ---------------------------------------
# CHECK IF FACES WERE LOADED
# ---------------------------------------

if len(known_face_encodings) == 0:

    print("\nNo registered faces found.")

    print("Run register.py first.")

    exit()


# ---------------------------------------
# START WEBCAM
# ---------------------------------------

camera = cv2.VideoCapture(0)

print("\nFace recognition started.")
print("Press Q to quit.\n")


# ---------------------------------------
# MAIN LOOP
# ---------------------------------------

while True:

    success, frame = camera.read()

    if not success:

        print("Could not access camera.")

        break


    # Make image smaller for faster processing
    small_frame = cv2.resize(
        frame,
        (0, 0),
        fx=0.25,
        fy=0.25
    )


    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        small_frame,
        cv2.COLOR_BGR2RGB
    )


    # Find faces
    face_locations = face_recognition.face_locations(
        rgb_frame
    )


    # Convert detected faces into encodings
    face_encodings = face_recognition.face_encodings(
        rgb_frame,
        face_locations
    )


    # Process every detected face
    for face_encoding, face_location in zip(
        face_encodings,
        face_locations
    ):

        # Compare current face with registered faces
        matches = face_recognition.compare_faces(
            known_face_encodings,
            face_encoding,
            tolerance=0.5
        )


        name = "Unknown"


        # Calculate face distances
        face_distances = face_recognition.face_distance(
            known_face_encodings,
            face_encoding
        )


        if len(face_distances) > 0:

            best_match_index = face_distances.argmin()


            if matches[best_match_index]:

                name = known_face_names[
                    best_match_index
                ]


        # Get face coordinates
        top, right, bottom, left = face_location


        # Convert coordinates back to original size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4


        # Draw rectangle around face
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )


        # Draw name background
        cv2.rectangle(
            frame,
            (left, bottom - 35),
            (right, bottom),
            (0, 255, 0),
            cv2.FILLED
        )


        # Display person's name
        cv2.putText(
            frame,
            name,
            (left + 6, bottom - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2
        )


    # Display final frame
    cv2.imshow(
        "Face Recognition",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ---------------------------------------
# CLEANUP
# ---------------------------------------

camera.release()

cv2.destroyAllWindows()