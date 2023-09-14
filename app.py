import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load the face recognition model and face detector
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_recognizer = cv2.createLBPHFaceRecognizer()
face_recognizer.load('trained_model.yml')

# Define the directory for storing uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define the threshold for face matching
MATCH_THRESHOLD = 70

# Define a route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Define a route for uploading an image
@app.route('/upload', methods=['POST'])
def upload():
    # Get the uploaded image file
    file = request.files['file']

    # Save the image to the uploads directory
    file.save(os.path.join(UPLOAD_FOLDER, file.filename))

    # Return a JSON response indicating success
    return jsonify({'success': True})

# Define a route for starting the exam
@app.route('/start_exam', methods=['POST'])
def start_exam():
    # Capture a frame from the user's webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # If no faces are detected, return a JSON response indicating failure
    if len(faces) == 0:
        return jsonify({'success': False, 'message': 'No faces detected.'})

    # Extract the first face detected
    x, y, w, h = faces[0]

    # Extract the face region from the frame
    face = gray[y:y+h, x:x+w]

    # Resize the face to match the size of the uploaded image
    uploaded_image = cv2.imread(os.path.join(UPLOAD_FOLDER, 'uploaded_image.jpg'))
    resized_face = cv2.resize(face, uploaded_image.shape[:2])

    # Predict the identity of the face using the trained model
    label, confidence = face_recognizer.predict(resized_face)

    # If the confidence is below the threshold, the faces match
    if confidence < MATCH_THRESHOLD:
        # Start the exam and proctoring
        return jsonify({'success': True, 'message': 'Exam started.'})
    else:
        # Display a message to the user that the faces do not match
        return jsonify({'success': False, 'message': 'Faces do not match.'})

if __name__ == '__main__':
    app.run(debug=True)
