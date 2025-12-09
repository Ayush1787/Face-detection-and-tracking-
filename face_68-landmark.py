import cv2 # type: ignore
import dlib # type: ignore
import numpy as np # type: ignore
from scipy.spatial import distance # type: ignore

class EmotionRecognizer:
    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat"):
        """
        Initialize the emotion recognizer with dlib's facial landmark detector
        Download the predictor from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        """
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        
    def get_landmarks(self, image):
        """Extract 68 facial landmarks from image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if len(faces) == 0:
            return None, None
            
        face = faces[0]
        landmarks = self.predictor(gray, face)
        
        # Convert to numpy array
        coords = np.zeros((68, 2), dtype=int)
        for i in range(68):
            coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
            
        return coords, face
    
    def eye_aspect_ratio(self, eye_points):
        """Calculate Eye Aspect Ratio (EAR)"""
        A = distance.euclidean(eye_points[1], eye_points[5])
        B = distance.euclidean(eye_points[2], eye_points[4])
        C = distance.euclidean(eye_points[0], eye_points[3])
        ear = (A + B) / (2.0 * C)
        return ear
    
    def mouth_aspect_ratio(self, mouth_points):
        """Calculate Mouth Aspect Ratio (MAR)"""
        A = distance.euclidean(mouth_points[13], mouth_points[19])  # 62-68
        B = distance.euclidean(mouth_points[14], mouth_points[18])  # 63-67
        C = distance.euclidean(mouth_points[15], mouth_points[17])  # 64-66
        D = distance.euclidean(mouth_points[12], mouth_points[16])  # 61-65
        mar = (A + B + C) / (2.0 * D)
        return mar
    
    def eyebrow_distance(self, landmarks):
        """Calculate distance between eyebrows (for anger/surprise)"""
        left_brow = landmarks[21]  # Left eyebrow center
        right_brow = landmarks[22]  # Right eyebrow center
        nose_bridge = landmarks[27]  # Nose bridge
        
        left_dist = distance.euclidean(left_brow, nose_bridge)
        right_dist = distance.euclidean(right_brow, nose_bridge)
        
        return (left_dist + right_dist) / 2.0
    
    def mouth_width_height_ratio(self, mouth_points):
        """Calculate mouth width to height ratio"""
        width = distance.euclidean(mouth_points[0], mouth_points[6])  # 49-55
        height = distance.euclidean(mouth_points[3], mouth_points[9])  # 52-58
        return width / height if height > 0 else 0
    
    def recognize_emotion(self, image):
        """
        Recognize emotion from facial landmarks
        Returns: emotion label and confidence
        """
        landmarks, face = self.get_landmarks(image)
        
        if landmarks is None:
            return "No face detected", 0.0
        
        # Extract facial regions
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        mouth = landmarks[48:68]
        
        # Calculate features
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        mar = self.mouth_aspect_ratio(mouth)
        brow_dist = self.eyebrow_distance(landmarks)
        mouth_ratio = self.mouth_width_height_ratio(mouth)
        
        # Simple rule-based emotion classification
        # These thresholds may need tuning based on your specific use case
        
        if mar > 0.6 and avg_ear > 0.25:
            emotion = "Surprised"
            confidence = min(mar * 1.2, 1.0)
        elif mouth_ratio > 3.5 and mar < 0.3:
            emotion = "Happy"
            confidence = min(mouth_ratio / 4.5, 1.0)
        elif mar > 0.4 and mouth_ratio < 2.5:
            emotion = "Sad"
            confidence = 0.7
        elif brow_dist < 25 and mar < 0.3:
            emotion = "Angry"
            confidence = 0.75
        elif avg_ear < 0.2:
            emotion = "Tired/Sleepy"
            confidence = 0.6
        elif mar < 0.25 and mouth_ratio < 2.8:
            emotion = "Neutral"
            confidence = 0.65
        else:
            emotion = "Neutral"
            confidence = 0.5
            
        return emotion, confidence
    
    def draw_landmarks(self, image, landmarks):
        """Draw facial landmarks on image"""
        for i, (x, y) in enumerate(landmarks):
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        return image


def main():
    """Example usage with webcam"""
    recognizer = EmotionRecognizer("shape_predictor_68_face_landmarks.dat")
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit, 's' to toggle landmarks display")
    show_landmarks = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get emotion
        emotion, confidence = recognizer.recognize_emotion(frame)
        
        # Get landmarks for visualization
        landmarks, face = recognizer.get_landmarks(frame)
        
        if landmarks is not None:
            # Draw face rectangle
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Draw landmarks if enabled
            if show_landmarks:
                frame = recognizer.draw_landmarks(frame, landmarks)
            
            # Display emotion
            text = f"{emotion} ({confidence:.2f})"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
        
        cv2.imshow('Emotion Recognition', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            show_landmarks = not show_landmarks
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()