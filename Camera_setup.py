import cv2
from ultralytics import YOLO
import numpy as np
import tkinter as tk
from tkinter import font as tkFont
from PIL import Image, ImageTk
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VehicleDetector:
    def __init__(self):
        # Load YOLOv8 model
        try:
            self.model = YOLO("yolov8n.pt")  # Nano model for speed
            logger.info("YOLOv8 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}")
            raise

        # Initialize Tkinter GUI
        self.root = tk.Tk()
        self.root.title("Vehicle Detection and Counting")
        self.root.geometry("800x480")  # Reduced window size

        # Left frame for video feed
        self.video_label = tk.Label(self.root)
        self.video_label.pack(side=tk.LEFT, padx=5, pady=5)

        # Right frame for info
        self.info_frame = tk.Frame(self.root)
        self.info_frame.pack(side=tk.RIGHT, padx=5, pady=5)

        # Info labels
        self.font_title = tkFont.Font(family="Helvetica", size=16, weight="bold")
        self.font_text = tkFont.Font(family="Helvetica", size=10)

        tk.Label(self.info_frame, text="Vehicle Detection", font=self.font_title).pack(pady=5)
        self.fps_label = tk.Label(self.info_frame, text="FPS: 0", font=self.font_text)
        self.fps_label.pack(pady=3)
        self.vehicle_count_label = tk.Label(self.info_frame, text="Vehicles Detected: 0", font=self.font_text)
        self.vehicle_count_label.pack(pady=3)
        self.status_label = tk.Label(self.info_frame, text="Status: Starting...", font=self.font_text)
        self.status_label.pack(pady=3)
        tk.Button(self.info_frame, text="Exit", command=self.root.quit).pack(pady=5)

        # FPS calculation variables
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0

        # Initialize camera with reduced resolution
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logger.error("Could not open camera")
            self.status_label.config(text="Error: Camera not accessible")
            raise Exception("Camera not accessible")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Target classes for vehicles (COCO dataset)
        self.target_classes = {
            2: "car",      # COCO class ID for car
            5: "bus",      # COCO class ID for bus
            7: "truck"     # COCO class ID for truck
        }
        
        # Frame skip counter
        self.frame_skip = 2  # Process every 2nd frame
        self.frame_counter = 0

    def process_frame(self):
        try:
            # Read frame from camera
            ret, frame = self.cap.read()
            if not ret:
                self.status_label.config(text="Error: Failed to capture frame")
                logger.error("Failed to capture frame")
                return

            self.frame_counter += 1
            if self.frame_counter % self.frame_skip != 0:  # Skip frames
                self.root.after(10, self.process_frame)
                return

            # Perform YOLO detection
            results = self.model(frame, conf=0.5)  # Increase confidence threshold
            vehicle_count = 0

            # Process detections
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in self.target_classes:
                        vehicle_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        label = f"{self.target_classes[cls_id]} {conf:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Thinner boxes
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Update vehicle count
            self.vehicle_count_label.config(text=f"Vehicles Detected: {vehicle_count}")

            # Calculate FPS
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 1:
                self.fps = self.frame_count / elapsed_time
                self.fps_label.config(text=f"FPS: {int(self.fps)}")
                self.frame_count = 0
                self.start_time = time.time()

            # Resize frame for display
            frame_resized = cv2.resize(frame, (480, 360))  # Smaller display size
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

            # Schedule next frame
            self.root.after(10, self.process_frame)

        except Exception as e:
            logger.error(f"Error in processing frame: {str(e)}")
            self.status_label.config(text=f"Error: {str(e)}")

    def run(self):
        try:
            self.status_label.config(text="Status: Running")
            self.process_frame()
            self.root.mainloop()
        except Exception as e:
            logger.error(f"Run error: {str(e)}")
            self.status_label.config(text=f"Error: {str(e)}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    try:
        detector = VehicleDetector()
        detector.run()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")