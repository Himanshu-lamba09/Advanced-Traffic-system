import cv2
import numpy as np
import time
import logging
import requests
import base64
import json
from ultralytics import YOLO
from pathlib import Path
from threading import Thread, Event, Lock
import os
import platform

# Mock GPIO for Windows, use real GPIO on Raspberry Pi
IS_RASPBERRY_PI = platform.system() == "Linux" and "raspberry" in platform.uname().machine.lower()
if IS_RASPBERRY_PI:
    import RPi.GPIO as GPIO
else:
    class DummyGPIO:
        def setmode(self, *args): pass
        def setup(self, *args): pass
        def output(self, *args): pass
        def cleanup(self): pass
        BCM = OUT = HIGH = LOW = None
    GPIO = DummyGPIO()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='traffic_system.log',
    filemode='w'
)

# Lane GPIO pins
lane_pins = {
    1: {'red': 2, 'yellow': 3, 'green': 4},
    2: {'red': 23, 'yellow': 24, 'green': 25},
    3: {'red': 17, 'yellow': 27, 'green': 22},
    4: {'red': 19, 'yellow': 26, 'green': 21}
}

# Initialize GPIO
if IS_RASPBERRY_PI:
    GPIO.setmode(GPIO.BCM)
    for lane in lane_pins.values():
        GPIO.setup(lane['red'], GPIO.OUT)
        GPIO.setup(lane['yellow'], GPIO.OUT)
        GPIO.setup(lane['green'], GPIO.OUT)

# Traffic light states
RED, YELLOW, GREEN = "Red", "Yellow", "Green"

# Configuration
CONFIG = {
    'min_switch_interval': 5,
    'yellow_duration': 3.0,
    'red_max_duration': 60.0,
    'frame_size': (320, 180),
    'base_fps': 30,
    'inference_interval': 3,
    'density_threshold_low': 2,
    'density_threshold_high': 5,
    'ambulance_confidence_threshold': 0.5,  # Increased from 0.1 to 0.5
}

# Roboflow API settings
API_KEY = "mkrQybLzJ5EcG1ppSfGr"
PROJECT = "ambulance-detection-ylj92"
VERSION = "2"
API_URL = f"https://detect.roboflow.com/{PROJECT}/{VERSION}?api_key={API_KEY}"

# Vehicle classes from COCO dataset (YOLOv8) to filter out humans
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}  # Exclude 0: "person"

class TrafficLane:
    def __init__(self, source, lane_id, is_camera=False):
        logging.debug(f"Initializing TrafficLane {lane_id} with source: {source}")
        self.is_camera = is_camera
        if is_camera:
            self.cap = cv2.VideoCapture(0)
        else:
            if not os.path.exists(source):
                logging.error(f"Video not found at {source}")
                raise ValueError(f"Video not found at {source}")
            self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            logging.error(f"Failed to open {'camera' if is_camera else 'video'}: {source}")
            raise ValueError(f"Failed to open {'camera' if is_camera else 'video'}: {source}")
        
        self.lane_id = lane_id
        self.model_vehicle = YOLO('yolov8n.pt', task='detect')
        self.frame = None
        self.vehicle_count = 0
        self.has_ambulance = False
        self.frame_count = 0
        self.stop_event = Event()
        self.lock = Lock()
        self.thread = Thread(target=self.process_frames, daemon=True)
        self.thread.start()

    def process_frames(self):
        target_frame_time = 1 / CONFIG['base_fps']
        logging.debug(f"Starting frame processing for Lane {self.lane_id}")
        
        while not self.stop_event.is_set():
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                if self.is_camera:
                    logging.error(f"Camera feed lost for Lane {self.lane_id}")
                    break
                else:
                    logging.warning(f"End of video for Lane {self.lane_id}, looping...")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

            self.frame_count += 1
            frame_resized = cv2.resize(frame, CONFIG['frame_size'], interpolation=cv2.INTER_NEAREST)
            
            if self.frame_count % CONFIG['inference_interval'] == 0:
                # Vehicle detection with filtering
                vehicle_results = self.model_vehicle(frame_resized, device='cpu')
                with self.lock:
                    # Filter detections to count only vehicles
                    self.vehicle_count = sum(
                        1 for box in vehicle_results[0].boxes
                        if int(box.cls) in VEHICLE_CLASSES
                    )
                
                # Ambulance detection with Roboflow API
                small_frame = cv2.resize(frame, (224, 224))
                _, buffer = cv2.imencode('.jpg', small_frame)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                headers = {"Content-Type": "application/x-www-form-urlencoded"}
                payload = img_base64
                
                try:
                    response = requests.post(API_URL, headers=headers, data=payload)
                    response.raise_for_status()
                    results = response.json()
                    predictions = results.get("predictions", [])
                    with self.lock:
                        self.has_ambulance = False
                        for pred in predictions:
                            confidence = pred.get("confidence", 0)
                            class_name = pred.get("class", "")
                            logging.debug(f"Lane {self.lane_id} - Prediction: {class_name}, Confidence: {confidence}")
                            if (class_name == "ambulance" and 
                                confidence >= CONFIG['ambulance_confidence_threshold']):
                                self.has_ambulance = True
                                annotated_frame = frame.copy()
                                x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                                scale_x = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 224
                                scale_y = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 224
                                x1, y1 = int((x - w/2) * scale_x), int((y - h/2) * scale_y)
                                x2, y2 = int((x + w/2) * scale_x), int((y + h/2) * scale_y)
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(annotated_frame, f"Ambulance {confidence:.2f}", 
                                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                output_path = f"ambulance_detected_lane_{self.lane_id}.jpg"
                                cv2.imwrite(output_path, annotated_frame)
                                logging.info(f"Ambulance detected in Lane {self.lane_id}, saved to {output_path}")
                                break
                except requests.exceptions.RequestException as e:
                    logging.error(f"API request failed for Lane {self.lane_id}: {e}")

            with self.lock:
                self.frame = frame_resized

            elapsed = time.time() - start_time
            time.sleep(max(0, target_frame_time - elapsed))

    def get_vehicle_count(self):
        with self.lock:
            return self.vehicle_count

    def has_ambulance_detected(self):
        with self.lock:
            return self.has_ambulance

    def get_frame(self):
        with self.lock:
            return self.frame if self.frame is not None else np.zeros((CONFIG['frame_size'][1], CONFIG['frame_size'][0], 3), dtype=np.uint8)

    def cleanup(self):
        logging.debug(f"Cleaning up Lane {self.lane_id}")
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.cap.release()

class TrafficManager:
    def __init__(self, sources):
        logging.debug("Initializing TrafficManager")
        self.lanes = [
            TrafficLane(sources[0], 1, is_camera=True),
            TrafficLane(sources[1], 2, is_camera=False),
            TrafficLane(sources[2], 3, is_camera=False),
            TrafficLane(sources[3], 4, is_camera=False)
        ]
        self.lights = [RED] * 4
        self.last_switch_times = [time.time()] * 4
        self.red_start_times = [time.time()] * 4
        self.red_timers = [0.0] * 4
        self.yellow_start_times = [None] * 4
        self.yellow_timers = [0.0] * 4
        self.current_green_lane = None
        self.sequence_index = 0

    def update_traffic_lights(self):
        current_time = time.time()
        vehicle_counts = [lane.get_vehicle_count() for lane in self.lanes]
        ambulance_lanes = [i for i, lane in enumerate(self.lanes) if lane.has_ambulance_detected()]
        
        for i in range(4):
            if self.red_start_times[i]:
                self.red_timers[i] = min(current_time - self.red_start_times[i], CONFIG['red_max_duration'])
            if self.yellow_start_times[i]:
                self.yellow_timers[i] = min(current_time - self.yellow_start_times[i], CONFIG['yellow_duration'])

        for i in range(4):
            if self.lights[i] == YELLOW and self.yellow_timers[i] >= CONFIG['yellow_duration']:
                self._switch_to_green(i, current_time)
                logging.info(f"Lane {i+1} switched from Yellow to Green")

        if ambulance_lanes:
            next_lane = ambulance_lanes[self.sequence_index % len(ambulance_lanes)]
            if self.current_green_lane != next_lane or self.lights[next_lane] != GREEN:
                self._switch_to_green(next_lane, current_time)
                self.current_green_lane = next_lane
                for i in range(4):
                    if i != next_lane:
                        self._switch_to_red(i, current_time)
            self.sequence_index += 1
            return self._get_status()

        density_scores = [
            self._calculate_density_score(count, i, current_time)
            for i, count in enumerate(vehicle_counts)
        ]
        max_score = max(density_scores)
        max_score_lanes = [i for i, score in enumerate(density_scores) if score == max_score and score > 0]

        if max_score > 0:
            next_lane = max_score_lanes[self.sequence_index % len(max_score_lanes)]
            if (self.current_green_lane != next_lane and 
                (self.lights[next_lane] == RED and 
                 self.red_timers[next_lane] >= CONFIG['min_switch_interval'])):
                self._start_transition_to_green(next_lane, current_time)
                self.current_green_lane = next_lane
                for i in range(4):
                    if i != next_lane:
                        self._switch_to_red(i, current_time)
            self.sequence_index += 1
        else:
            for i in range(4):
                if self.lights[i] != RED:
                    self._switch_to_red(i, current_time)

        return self._get_status()

    def _calculate_density_score(self, vehicle_count, lane_idx, current_time):
        base_score = min(vehicle_count, CONFIG['density_threshold_high'])
        if vehicle_count >= CONFIG['density_threshold_high']:
            base_score *= 1.5
        elif vehicle_count <= CONFIG['density_threshold_low']:
            base_score *= 0.5
        
        if self.lights[lane_idx] == RED:
            red_time_factor = min(self.red_timers[lane_idx] / CONFIG['red_max_duration'], 1.0)
            base_score += base_score * red_time_factor
        return base_score

    def _start_transition_to_green(self, lane_idx, current_time):
        if self.lights[lane_idx] != YELLOW:
            self.lights[lane_idx] = YELLOW
            self.yellow_start_times[lane_idx] = current_time
            self.yellow_timers[lane_idx] = 0.0
            self.red_start_times[lane_idx] = None
            self.red_timers[lane_idx] = 0.0
            self.last_switch_times[lane_idx] = current_time
            if IS_RASPBERRY_PI:
                GPIO.output(lane_pins[lane_idx + 1]['red'], GPIO.LOW)
                GPIO.output(lane_pins[lane_idx + 1]['yellow'], GPIO.HIGH)
                GPIO.output(lane_pins[lane_idx + 1]['green'], GPIO.LOW)
            logging.info(f"Lane {lane_idx+1} switched to Yellow")

    def _switch_to_green(self, lane_idx, current_time):
        self.lights[lane_idx] = GREEN
        self.red_start_times[lane_idx] = None
        self.red_timers[lane_idx] = 0.0
        self.yellow_start_times[lane_idx] = None
        self.yellow_timers[lane_idx] = 0.0
        self.last_switch_times[lane_idx] = current_time
        if IS_RASPBERRY_PI:
            GPIO.output(lane_pins[lane_idx + 1]['red'], GPIO.LOW)
            GPIO.output(lane_pins[lane_idx + 1]['yellow'], GPIO.LOW)
            GPIO.output(lane_pins[lane_idx + 1]['green'], GPIO.HIGH)
        logging.info(f"Lane {lane_idx+1} switched to Green")

    def _switch_to_red(self, lane_idx, current_time):
        if self.lights[lane_idx] != RED:
            self.lights[lane_idx] = RED
            self.red_start_times[lane_idx] = current_time
            self.red_timers[lane_idx] = 0.0
            self.yellow_start_times[lane_idx] = None
            self.yellow_timers[lane_idx] = 0.0
            self.last_switch_times[lane_idx] = current_time
            if IS_RASPBERRY_PI:
                GPIO.output(lane_pins[lane_idx + 1]['red'], GPIO.HIGH)
                GPIO.output(lane_pins[lane_idx + 1]['yellow'], GPIO.LOW)
                GPIO.output(lane_pins[lane_idx + 1]['green'], GPIO.LOW)
            logging.info(f"Lane {lane_idx+1} switched to Red")

    def _get_status(self):
        status = f"Lane 1: {self.lights[0]}, Lane 2: {self.lights[1]}, Lane 3: {self.lights[2]}, Lane 4: {self.lights[3]}"
        red_timers_display = [f"{int(t)}" if t > 0 else "0" for t in self.red_timers]
        return status, red_timers_display

    def cleanup(self):
        logging.debug("Cleaning up TrafficManager")
        for lane in self.lanes:
            lane.cleanup()
        if IS_RASPBERRY_PI:
            GPIO.cleanup()

def main():
    sources = [
        0,  # Lane 1: Live camera feed
        "C:\\Users\\user\\OneDrive\\Desktop\\traffic system\\road2.mp4",
        "C:\\Users\\user\\OneDrive\\Desktop\\traffic system\\empty.mp4",
        "C:\\Users\\user\\OneDrive\\Desktop\\traffic system\\road2.mp4"
    ]
    video_paths = sources[1:]
    if not all(Path(p).exists() for p in video_paths):
        logging.error("One or more video files not found: %s", video_paths)
        print("Error: Check video files exist in the project directory")
        return

    traffic_mgr = TrafficManager(sources)
    logging.info("TrafficManager initialized successfully")

    screen_width, screen_height = 1280, 720
    target_width = screen_width // 2
    target_height = screen_height // 2
    timer_height = 50
    time.sleep(1)

    while True:
        frames = [lane.get_frame() for lane in traffic_mgr.lanes]
        
        light_status, red_timers = traffic_mgr.update_traffic_lights()
        vehicle_counts = [lane.get_vehicle_count() for lane in traffic_mgr.lanes]

        resized_frames = []
        for i, frame in enumerate(frames):
            resized_frame = cv2.resize(frame, (target_width, target_height - timer_height), interpolation=cv2.INTER_NEAREST)
            timer_canvas = np.zeros((timer_height, target_width, 3), dtype=np.uint8)

            total_vehicles = vehicle_counts[i]
            cv2.putText(resized_frame, f"Vehicles: {total_vehicles}", 
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            light_color = (0, 255, 0) if traffic_mgr.lights[i] == GREEN else \
                         (0, 255, 255) if traffic_mgr.lights[i] == YELLOW else (0, 0, 255)
            cv2.putText(resized_frame, f"Light: {traffic_mgr.lights[i]}", 
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, light_color, 2)
            if traffic_mgr.lanes[i].has_ambulance_detected():
                cv2.putText(resized_frame, "Ambulance!", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if traffic_mgr.lights[i] == RED:
                cv2.putText(timer_canvas, f"Red: {red_timers[i]}s", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif traffic_mgr.lights[i] == YELLOW:
                cv2.putText(timer_canvas, f"Yellow: {int(traffic_mgr.yellow_timers[i])}s", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(timer_canvas, "Green", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            combined_frame = np.vstack((resized_frame, timer_canvas))
            resized_frames.append(combined_frame)

        top_row = np.hstack((resized_frames[0], resized_frames[1]))
        bottom_row = np.hstack((resized_frames[2], resized_frames[3]))
        combined_frame = np.vstack((top_row, bottom_row))
        cv2.putText(combined_frame, light_status, (10, combined_frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Traffic Management System - 4 Lanes", combined_frame)
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break

    traffic_mgr.cleanup()
    cv2.destroyAllWindows()
    logging.info("Program terminated")

if __name__ == "__main__":
    main() 