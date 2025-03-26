import cv2
import os
import requests
import base64
import json

# Replace these with your Roboflow credentials
API_KEY = "uU5h1GPFJ8YF2k22Lglj"
WORKSPACE = "https://app.roboflow.com/ambulance-poag1"
PROJECT = "https://app.roboflow.com/ambulance-poag1/ambulance-detection-ylj92"
VERSION = "2"  # Replace with your model version number
API_URL = f"https://detect.roboflow.com/ambulance-detection-ylj92/2"

def process_ambulance_video(video_path, output_photo_path="ambulance_detected.jpg"):
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame for inference (maintaining aspect ratio)
        small_frame = cv2.resize(frame, (224, 224))
        
        # Convert frame to base64 for API
        _, buffer = cv2.imencode('.jpg', small_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare API request
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "api_key": API_KEY,
            "image": img_base64,
            "confidence": 0.1,  # Low confidence to catch more detections
            "overlap": 0.45     # Less strict IoU (equivalent to NMS)
        }
        
        # Send request to Roboflow API
        try:
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # Raise exception for bad status codes
            results = response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            continue
        
        # Parse detections from API response
        predictions = results.get("predictions", [])
        ambulance_detected = False
        
        for prediction in predictions:
            class_name = prediction.get("class")
            confidence = prediction.get("confidence")
            if class_name == "ambulance" and confidence >= 0.1:  # Adjust class name as per your model
                ambulance_detected = True
                # Draw bounding box on the original frame
                x, y, w, h = prediction["x"], prediction["y"], prediction["width"], prediction["height"]
                # Scale coordinates back to original frame size
                x1 = int((x - w / 2) * frame_width / 224)
                y1 = int((y - h / 2) * frame_height / 224)
                x2 = int((x + w / 2) * frame_width / 224)
                y2 = int((y + h / 2) * frame_height / 224)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Ambulance {confidence:.2f}", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                break
        
        # Display the frame
        cv2.imshow('Ambulance Detection', frame)
        
        frame_count += 1
        
        # If ambulance detected, save photo and stop
        if ambulance_detected:
            print(f"Ambulance DETECTED at frame {frame_count}")
            print(f"Saving photo to {output_photo_path}")
            cv2.imwrite(output_photo_path, frame)
            break
        
        # Print status every 30 frames if no ambulance detected yet
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: No ambulance detected yet")
        
        # Press 'q' to quit manually
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release everything
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final status
    if ambulance_detected:
        print(f"Processing stopped - Ambulance detected and photo saved")
    else:
        print(f"Processing complete - No ambulance detected in {frame_count} frames")

# Main function
if __name__ == "__main__":
    # Replace with your actual video path
    video_path = "ambulance.mp4"
    process_ambulance_video(video_path)