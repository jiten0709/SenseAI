import cv2
import face_recognition
import os
import pickle
import csv
from datetime import datetime

# Directory to store face encodings
FACE_DIR = "faces"
os.makedirs(FACE_DIR, exist_ok=True)

# CSV file to store attendance records
ATTENDANCE_FILE = "attendance.csv"

# Register a new face
def register_face(name):
    cam = cv2.VideoCapture(0)
    print("Press 'q' to capture the face.")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        cv2.imshow("Register Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            face_locations = face_recognition.face_locations(frame)
            if face_locations:
                face_encoding = face_recognition.face_encodings(frame, face_locations)[0]
                with open(f"{FACE_DIR}/{name}.pkl", "wb") as f:
                    pickle.dump(face_encoding, f)
                print(f"Face for {name} registered successfully.")
                break
            else:
                print("No face detected. Try again.")

    cam.release()
    cv2.destroyAllWindows()

# Load known faces
def load_faces():
    faces = {}
    for file in os.listdir(FACE_DIR):
        name = os.path.splitext(file)[0]
        with open(f"{FACE_DIR}/{file}", "rb") as f:
            faces[name] = pickle.load(f)
    return faces

# Mark attendance
def mark_attendance(name):
    with open(ATTENDANCE_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    print(f"Attendance marked for {name}.")

# Recognize faces and mark attendance
def recognize_faces():
    known_faces = load_faces()
    cam = cv2.VideoCapture(0)
    print("Press 'q' to stop face recognition.")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding)
            if True in matches:
                name = list(known_faces.keys())[matches.index(True)]
                print(f"Recognized: {name}")
                mark_attendance(name)

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

# Main menu
def main():
    while True:
        print("\n1. Register Face")
        print("2. Mark Attendance")
        print("3. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            name = input("Enter your name: ")
            register_face(name)
        elif choice == "2":
            recognize_faces()
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
