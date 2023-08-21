import cv2
import dlib
from scipy.spatial import distance

# Calculate the eye aspect ratio (EAR) to detect blinks
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Constants for eye aspect ratio and blink detection
EAR_THRESHOLD = 0.3
BLINK_CONSEC_FRAMES = 3

# Load face and eye detectors
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Open the webcam
cap = cv2.VideoCapture(0)

# Initialize counters and state variables
blink_counter = 0
drunk_driving_detected = False

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = detector(gray)
    
    # Iterate over detected faces
    for face in faces:
        # Predict facial landmarks
        landmarks = predictor(gray, face)
        
        # Extract the left and right eye coordinates
        left_eye = []
        right_eye = []
        
        for n in range(36, 42):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            left_eye.append((x, y))
        for n in range(42, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            right_eye.append((x, y))
        
        # Calculate eye aspect ratios
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        
        # Average the eye aspect ratios
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Check if the average eye aspect ratio is below the threshold
        if avg_ear < EAR_THRESHOLD:
            blink_counter += 1
            if blink_counter >= BLINK_CONSEC_FRAMES:
                drunk_driving_detected = True
                break
        else:
            blink_counter = 0
    
    # Display the frame with eye landmarks
    for eye in [left_eye, right_eye]:
        for (x, y) in eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    
    # Display warning if drunk driving is detected
    if drunk_driving_detected:
        cv2.putText(frame, "DRUNK DRIVING DETECTED!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display the resulting frame
    cv2.imshow("Drunk Driving Detection", frame)
    
    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
