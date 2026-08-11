import cv2
import face_recognition
import pickle
import os


name = input("Enter person's name: ").strip()

if not name:
    print("Name cannot be empty.")
    exit()


cap = cv2.VideoCapture(0)

print("Camera started.")
print("Look at the camera and press SPACE to capture.")
print("Press Q to quit.")


while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not access camera.")
        break

    cv2.imshow("VisionID - Registration", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)

        if len(face_locations) == 0:
            print("No face detected. Try again.")

        elif len(face_locations) > 1:
            print("Multiple faces detected. Keep only one person in frame.")

        else:
            face_encoding = face_recognition.face_encodings(
                rgb_frame,
                face_locations
            )[0]

            os.makedirs("encodings", exist_ok=True)

            file_path = f"encodings/{name}.pkl"

            with open(file_path, "wb") as file:
                pickle.dump(face_encoding, file)

            print(f"Face registered successfully for {name}")
            print(f"Encoding saved to: {file_path}")

            break

    elif key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()