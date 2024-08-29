import os
import face_recognition
import numpy as np
import joblib
from sklearn import neighbors

def load_images_from_folder(folder):
    images = []
    labels = []
    for subdir in os.listdir(folder):
        subfolder = os.path.join(folder, subdir)
        if os.path.isdir(subfolder):
            for filename in os.listdir(subfolder):
                img_path = os.path.join(subfolder, filename)
                img = face_recognition.load_image_file(img_path)
                img_encoding = face_recognition.face_encodings(img)
                if img_encoding:
                    images.append(img_encoding[0])
                    labels.append(subdir)
    return images, labels

# Load images and labels
folder_path = 'training_face'  # Thay đổi đường dẫn đến thư mục chứa ảnh
images, labels = load_images_from_folder(folder_path)

# Train KNN classifier
knn_clf = neighbors.KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree', weights='distance')
knn_clf.fit(images, labels)

# Save the trained model
joblib.dump(knn_clf, 'Deploy_AI_Model\knn_model.pkl')
