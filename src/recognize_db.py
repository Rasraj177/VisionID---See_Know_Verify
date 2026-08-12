import cv2
import face_recognition
import pickle

from database import get_connection


# Connect to database
connection = get_connection()
cursor = connection.cursor()

# Get registered users
cursor.execute(
    "SELECT id, name, face_encoding FROM users"
)

users = cursor.fetchall()

cursor.close()
connection.close()


# Store encodings and names
known_encodings = []
known_names = []
known_ids = []


for user_id, name, encoding_data in users:

    encoding = pickle.loads(encoding_data)

    known_ids.append(user_id)
    known_names.append(name)
    known_encodings.append(encoding)


print("Registered users:", known_names)


# Start webcam
cap = cv2.VideoCapture(0)

print("Camera started.")
print("Press Q to quit.")


while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not access camera.")
        break


    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Detect faces
    face_locations = face_recognition.face_locations(
        rgb_frame
    )


    # Generate embeddings
    face_encodings = face_recognition.face_encodings(
        rgb_frame,
        face_locations
    )


    # Process every detected face
    for face_encoding, location in zip(
        face_encodings,
        face_locations
    ):

        name = "Unknown"
        user_id = None


        if known_encodings:

            # Compare face with registered faces
            matches = face_recognition.compare_faces(
                known_encodings,
                face_encoding,
                tolerance=0.6
            )


            # Calculate distances
            face_distances = face_recognition.face_distance(
                known_encodings,
                face_encoding
            )


            best_match_index = face_distances.argmin()


            # Check best match
            if matches[best_match_index]:

                name = known_names[
                    best_match_index
                ]

                user_id = known_ids[
                    best_match_index
                ]


        # Face coordinates
        top, right, bottom, left = location


        # Draw rectangle
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )


        # Display name
        label = name

        if user_id is not None:

            label = f"{name} (ID: {user_id})"


        cv2.putText(
            frame,
            label,
            (left, bottom + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # Display camera
    cv2.imshow(
        "VisionID - Database Recognition",
        frame
    )


    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()