import cv2
import os
import numpy as np
from PIL import Image

# Path for face image database
path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if not f.startswith('.')]
    faceSamples = []
    ids = []
    
    # We will use a simple dictionary to map names to numbers
    name_map = {}
    current_id = 0

    for imagePath in imagePaths:
        # Get the name from the folder name
        name = os.path.basename(imagePath)
        if name not in name_map:
            name_map[name] = current_id
            current_id += 1
        
        # Get all images in the subfolder
        for img_file in os.listdir(imagePath):
            if img_file.endswith(".jpg"):
                full_img_path = os.path.join(imagePath, img_file)
                PIL_img = Image.open(full_img_path).convert('L') # convert to grayscale
                img_numpy = np.array(PIL_img, 'uint8')
                
                faces = detector.detectMultiScale(img_numpy)
                for (x,y,w,h) in faces:
                    faceSamples.append(img_numpy[y:y+h,x:x+w])
                    ids.append(name_map[name])

    return faceSamples, ids

print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
if not os.path.exists('trainer'):
    os.makedirs('trainer')
recognizer.write('trainer/trainer.yml')

print(f"\n [INFO] {len(np.unique(ids))} faces trained. Exiting Program")