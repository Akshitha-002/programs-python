import cv2
import numpy as np
import face_recognition
import os
import random
from datetime import datetime

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            dString = now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{dtString},{dString}')

# --- 1. STUDENT DATABASE ---
# Store specific info for each student
student_info = {
    "AKSHITHA": {"class": "CSE-1", "reg": "2026001"},
    "ASHWITHA": {"class": "AIML-1", "reg": "2026002"},
    "ARBAB": {"class": "CSE-1", "reg": "2026003"},
    "BHOOMIKA": {"class": "CSE-1", "reg": "2026004"},
    "CHAITHANYA": {"class": "CSE-1", "reg": "2026005"}

}

path = 'Resources'
images, known_names, present_students = [], [], set()
myList = os.listdir(path)

# Load images and names from Resources folder
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    known_names.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete. Camera Starting...')

# --- 2. MATRIX SETTINGS ---
width, height = 1280, 720
font_size = 20
columns = int(width / font_size)
drops = [0 for _ in range(columns)]

cap = cv2.VideoCapture(0)

while True:
    # --- 3. BINARY BACKGROUND ---
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(len(drops)):
        char = str(random.randint(0, 1))
        x, y = i * font_size, drops[i] * font_size
        cv2.putText(bg, char, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        if y > height or random.random() > 0.95: drops[i] = 0
        else: drops[i] += 1
# ... after your binary background code ...
    
    # 2. Add Heading in the black space
    cv2.putText(bg, "SMART ATTENDANCE SYSTEM", (700, 50), 
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 3)

    # 3. Process Camera Feed
    success, img = cap.read()
    # ... existing camera overlay code ...
    # --- 4. RECOGNITION ---
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25) # Faster processing
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    
    # Overlay camera on background
    bg[50:530, 50:690] = cv2.resize(img, (640, 480))

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = known_names[matchIndex].upper()
            markAttendance(name)
            present_students.add(name) # Counter increases for unique students
            
            # --- 5. GREEN BOX AND DYNAMIC INFO ---
            y1, x2, y2, x1 = [v * 4 for v in faceLoc] # Scale up
            # Draw green box
            cv2.rectangle(bg, (x1+50, y1+50), (x2+50, y2+50), (0, 255, 0), 2)
            cv2.putText(bg, name, (x1+50, y1+45), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)


             # Display Details in the black space on the right
            if name in student_info:
                info = student_info[name]
                cv2.putText(bg, f"STUDENT: {name}", (700, 150), 1, 1.5, (255, 255, 255), 2)
                cv2.putText(bg, f"CLASS: {info['class']}", (700, 210), 1, 1.5, (255, 255, 255), 2)
                cv2.putText(bg, f"REG NO: {info['reg']}", (700, 270), 1, 1.5, (255, 255, 255), 2)
    # --- 6. ONE-BY-ONE COUNTER ---
    cv2.putText(bg, f"TOTAL PRESENT: {len(present_students)}", (720, 650), 1, 2, (0, 255, 0), 2)

    cv2.imshow('Attendance Dashboard', bg)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()