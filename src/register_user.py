import cv2
import face_recognition
import pickle

from database import get_connection


name = input("Enter name: ").strip()
email = input("Enter email: ").strip()
student_id = input("Enter student ID: ").strip()


if not name or not student_id:
    print("Name and student ID are required.")
    exit()


cap = cv2.VideoCapture(0)

print("Camera started.")
print("Press SPACE to capture your face.")
print("Press Q to quit.")


while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not access camera.")
        break


    cv2.imshow(
        "VisionID - Registration",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    if key == ord(" "):

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        face_locations = face_recognition.face_locations(
            rgb_frame
        )


        if len(face_locations) == 0:
            print("No face detected. Try again.")
            continue


        if len(face_locations) > 1:
            print("Multiple faces detected. Keep only one person.")
            continue


        encoding = face_recognition.face_encodings(
            rgb_frame,
            face_locations
        )[0]


        encoding_data = pickle.dumps(encoding)


        connection = get_connection()
        cursor = connection.cursor()


        query = """
        INSERT INTO users
        (name, email, student_id, face_encoding)
        VALUES (%s, %s, %s, %s)
        """


        values = (
            name,
            email,
            student_id,
            encoding_data
        )


        cursor.execute(
            query,
            values
        )


        connection.commit()


        cursor.close()
        connection.close()


        print("User registered successfully.")
        print(f"Name: {name}")
        print(f"Student ID: {student_id}")


        break


    elif key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()