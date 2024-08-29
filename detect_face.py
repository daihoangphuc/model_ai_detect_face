import cv2
import face_recognition
import joblib

# Load the trained KNN model
knn_clf = joblib.load('knn_model.pkl')

# Open the camera
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    # Convert the image from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the image
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Predict the label (name) of the face
        matches = knn_clf.predict([face_encoding])
        name = matches[0]

        # Draw rectangle around the face and write the name
        (top, right, bottom, left) = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit the program when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
video_capture.release()
cv2.destroyAllWindows()
