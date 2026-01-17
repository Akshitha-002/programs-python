from flask import Flask, render_template, Response
import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
from flask import Flask, render_template, Response, jsonify
# ... other imports ...

app = Flask(__name__)

# ADD THESE TWO LINES HERE
present_students = set()
last_detected = "Scanning..."

# 1. Initialize Flask at the top
app = Flask(__name__)

# 2. Configuration and Image Loading
print("Step 1: Starting the script...")
path = 'Images'
images = []
classNames = []
myList = os.listdir(path)
print(f"Step 2: Found these files in Images folder: {myList}")

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is None:
        print(f"--- ERROR: Could not read image {cl} ---")
    else:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
print(f"Step 3: Successfully loaded images for: {classNames}")

# 3. Encoding Function
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

print("Step 4: Starting face encoding (this may take a minute)...")
encodeListKnown = findEncodings(images)
print("Step 5: Encodings complete!")

# 4. Attendance Function
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            
        if name not in nameList:
            now = datetime.now()
            # Capture both Time and Date
            tString = now.strftime('%H:%M:%S')
            dString = now.strftime('%d-%m-%Y') 
            
            # Write Name, Time, and Date to a new line
            f.writelines(f'\n{name},{tString},{dString}')
            print(f"Success! Logged {name} on {dString}")

# 5. Video Generation
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        if not success:
            break
        else:
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    # ADD THESE TWO LINES HERE
                    last_detected = name
                    present_students.add(name)
                    markAttendance(name)
                    
                    # --- ADD DRAWING CODE HERE ---
                    y1, x2, y2, x1 = faceLoc
                    # Scale back up because we processed at 0.25 size
                    y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# 6. Web Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
# 7. Start Command (Must be at the very bottom and NOT indented)
@app.route('/get_stats')
def get_stats():
    return {
        "total_count": len(present_students),
        "last_name": last_detected
    }
if __name__ == "__main__":
    print("Step 6: Flask server starting now at http://127.0.0.1:5000")
    app.run(debug=True)