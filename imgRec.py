import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Use a video file instead of webcam
video_capture = cv2.VideoCapture("test_video.mp4")

# Load the known face image and get encoding
divansh_image = face_recognition.load_image_file("faces/goyal.jpeg")
divansh_encoding = face_recognition.face_encodings(divansh_image)
if len(divansh_encoding) == 0:
    raise ValueError("No face found in the reference image.")
divansh_encoding = divansh_encoding[0]  # Get the first face encoding

known_face_encoding = [divansh_encoding]
known_face_name = ["Divanshu"]

students = known_face_name.copy()

# Prepare CSV file to save attendance
now = datetime.now()
current_date = now.strftime("%y-%m-%d")
f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("End of video or cannot fetch the frame.")
        break

    # Resize frame for faster processing
    smallframe = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_smallframe = cv2.cvtColor(smallframe, cv2.COLOR_BGR2RGB)

    # Detect faces and get encodings
    face_locations = face_recognition.face_locations(rgb_smallframe)
    face_encodings = face_recognition.face_encodings(rgb_smallframe, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encoding, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_name[best_match_index]
            if name in students:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 100)
                fontScale = 1.5
                fontColor = (255, 0, 0)
                thickness = 3
                lineType = 2
                cv2.putText(frame, name + " present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

                students.remove(name)
                current_time = datetime.now().strftime("%H:%M:%S")
                lnwriter.writerow([name, current_time])

    cv2.imshow("Attendance", frame)

    # Exit when 'q' key pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
f.close()
