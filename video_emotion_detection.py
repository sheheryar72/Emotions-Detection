from deepface import DeepFace
import cv2
# from google.colab.patches import cv2_imshow

video_path = "./videos/video6.mp4"

face_model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

capture = cv2.VideoCapture(video_path)

frame_list = []

while True:
    ret, frame = capture.read()
    if not ret:
        break  

    faces = face_model.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 5)

    for x, y, width, height in faces:
        
        emotions = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        for emotion in emotions:
            emotion_label = emotion["dominant_emotion"]
            
            cv2.putText(frame, str(emotion_label), 
                        (x, y + height),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.9,
                        (255, 255, 0),
                        2)
        
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 0), 2)

    frame_list.append(frame)


height, width, colors = frame_list[0].shape
size = (width, height)

output_path = "Emotions.avi"
output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"DIVX"), 30, size)

for frame in frame_list:
    output.write(frame)


output.release()

print("Video processing complete.")


