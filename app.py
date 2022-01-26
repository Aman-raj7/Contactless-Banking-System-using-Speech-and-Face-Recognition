# Import necessary Library
from os import name
from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
import shutil

app=Flask(__name__)
camera = cv2.VideoCapture(0) # video capture object for the camera.

# Load a picture and learn how to recognize it.
Yogesh_image = face_recognition.load_image_file("Dataset/Yogesh/Yogesh.jpg")
Yogesh_face_encoding = face_recognition.face_encodings(Yogesh_image)[0]

# Load a second picture and learn how to recognize it.
Aman_raj_gupta_image = face_recognition.load_image_file("Dataset/Aman raj gupta/Aman Raj Gupta.jpeg")
Aman_raj_gupta_face_encoding = face_recognition.face_encodings(Aman_raj_gupta_image)[0]

# Load a Third picture and learn how to recognize it.
Sudhanshu_image = face_recognition.load_image_file("Dataset/Sudhanshu/Sudhanshu.jpg")
Sudhanshu_face_encoding = face_recognition.face_encodings(Sudhanshu_image)[0]

# Load a Fourth picture and learn how to recognize it.
Shivkumar_image = face_recognition.load_image_file("Dataset/Shivkumar/Shivkumar.jpg")
Shivkumar_face_encoding = face_recognition.face_encodings(Shivkumar_image)[0]

# Load a fifth picture and learn how to recognize it.
Pragya_image = face_recognition.load_image_file("Dataset/Pragya/Pragya.jpeg")
Pragya_face_encoding = face_recognition.face_encodings(Pragya_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    Yogesh_face_encoding,
    Aman_raj_gupta_face_encoding,
    Sudhanshu_face_encoding,
    Shivkumar_face_encoding,
    Pragya_face_encoding
]
known_face_names = [
    "Yogesh",
    "Aman raj Gupta",
    "Sudhanshu",
    "Shivkumar",
    "Pragya"
]
# Initialize some variables
face_locations = []
face_encodings = []

face_names  = []
process_this_frame = True


def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
           
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                
                
                face_names.append(name)
            

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                # save a user name in text file
                file1 = open("myfile.txt","w")#write mode
                file1.write(name)
                file1.close()   

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#Copy the file content to verify
original = r'myfile.txt'
target = r'E:\Academics\SEM_7\1 Project\Final\Tutorial 1-2\templates\file.txt'
shutil.copyfile(original, target)


@app.route('/')
def index():
    return render_template('Index.html') 

@app.route('/ind')
def ind():
    return render_template('Index.html') 

@app.route('/index2')
def index2():
    return render_template('index2.html')

@app.route('/choose')
def choose():
    return render_template('1choose.html')

@app.route('/withdraw')
def withdraw():
    return render_template('withdraw.html')

@app.route('/withdraw2')
def withdraw2():
    return render_template('withdraw2.html')

@app.route('/sheet')
def sheet():
    return render_template('sheet.css')

@app.route('/trans')
def trans():
    return render_template('trans.html')

@app.route('/trans2')
def trans2():
    return render_template('trans2.html')

@app.route('/error')
def error():
    return render_template('error.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__=='__main__':
    app.run(debug=True)
camera.release()