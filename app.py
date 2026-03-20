import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# 1. Load the AI Model
# 'trainer.yml' must be in a folder named 'trainer'
recognizer = cv2.face.LBPHFaceRecognizer_create()
if os.path.exists('trainer/trainer.yml'):
    recognizer.read('trainer/trainer.yml')

# Load the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 2. Your Student List
# ID 0 is 'None', ID 1 is the first person you trained (Gauri)
# Change this line in app.py
names = ['gauri', 'Unknown']

# --- VIDEO RECOGNITION LOGIC ---
def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # --- ADD THIS LINE TO FLIP THE IMAGE ---
        # 1 means horizontal flip (Mirror effect)
        frame = cv2.flip(frame, 1) 
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 80:
                name = names[id]
                # --- LOG ATTENDANCE ---
                from datetime import datetime
                now = datetime.now()
                dtString = now.strftime('%Y-%m-%d %H:%M:%S')
                with open('attendance.csv', 'a') as f:
                    f.writelines(f'\n{name},{dtString}')
                
                accuracy = f"  {round(100 - confidence)}%"
                color = (0, 255, 0) 
            else:
                name = "Unknown"
                accuracy = f"  {round(100 - confidence)}%"
                color = (0, 0, 255) 

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, accuracy, (x+5, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')#
# --- PAGE ROUTES (Fixed Structure) ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    attendance_data = []
    if os.path.exists('attendance.csv'):
        with open('attendance.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row: # Check if the row isn't empty
                    attendance_data.append(row)
    
    # Reverse the list so the newest attendance is at the top
    attendance_data.reverse() 
    return render_template('dashboard.html', students=attendance_data)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- DATASET COLLECTION ROUTE ---
@app.route('/start_collection/<username>')
def start_collection(username):
    user_path = f"dataset/{username}"
    if not os.path.exists(user_path):
        os.makedirs(user_path)
    
    camera = cv2.VideoCapture(0)
    count = 0
    while count < 20:
        success, frame = camera.read()
        if success:
            cv2.imwrite(f"{user_path}/sample_{count}.jpg", frame)
            count += 1
            cv2.waitKey(100)
    camera.release()
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(debug=True)