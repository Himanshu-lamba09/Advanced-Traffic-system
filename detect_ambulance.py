from ultralytics import YOLO
import cv2
import os

def process_ambulance_video(video_path, output_photo_path="ambulance_detected.jpg"):
    # Load the trained model
    model = YOLO("runs/detect/ambulance_fast_cpu/weights/best.pt")
    
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
        
        # Perform inference
        results = model.predict(
            source=small_frame,
            conf=0.1,        # Low confidence to catch more detections
            iou=0.45,        # Less strict IoU
            verbose=False    # Reduce console clutter
        )
        
        # Check detections
        detections = results[0].boxes
        detected_classes = [int(cls) for cls in detections.cls]
        ambulance_detected = False
        
        for box in detections:
            cls_id = int(box.cls)
            if cls_id == 0:  # Assuming 0 is 'ambulance'
                ambulance_detected = True
                break
        
        # Get annotated frame and resize back to original size
        annotated_frame = results[0].plot()
        annotated_frame = cv2.resize(annotated_frame, (frame_width, frame_height))
        
        # Display the frame
        cv2.imshow('Ambulance Detection', annotated_frame)
        
        frame_count += 1
        
        # If ambulance detected, save photo and stop
        if ambulance_detected:
            print(f"Ambulance DETECTED at frame {frame_count}")
            print(f"Saving photo to {output_photo_path}")
            cv2.imwrite(output_photo_path, annotated_frame)
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