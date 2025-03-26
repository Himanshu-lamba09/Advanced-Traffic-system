🚦 Smart Traffic Management System
📌 Project Overview
This project is a Smart Traffic Management System designed to dynamically control traffic signals based on real-time vehicle density and emergency vehicle detection. Using OpenCV for vehicle detection, TensorFlow (Roboflow) for ambulance recognition, and a custom traffic light control algorithm, this system ensures efficient traffic flow and prioritizes ambulances for emergency situations.

🔹 Key Features
✅ Real-Time Vehicle Detection – Uses OpenCV to count vehicles in each lane.
✅ Dynamic Traffic Light Control – The lane with the highest vehicle density gets the green signal.
✅ Ambulance Detection – If an ambulance is detected in any lane, that lane turns green immediately.
✅ Countdown Timer – A 10-second yellow light precedes green transitions.
✅ Adaptive Signal Timing – Red lights last 60 seconds, but can change earlier if traffic density increases.
✅ Live Traffic Data Display – Vehicle count and signal status are displayed below each camera feed.

🛠️ Hardware & Software Requirements
🔧 Hardware Components
Computer with Camera Feeds (for video input)

Raspberry Pi / Jetson Nano / PC (for processing, optional for deployment)

Traffic Light System (Simulated or Physical LEDs/Relays)

🖥️ Software Dependencies
Ensure you have the following installed:

Python 3.8+

OpenCV (for image processing and vehicle detection)

TensorFlow / Roboflow Model (for ambulance detection)

NumPy & Pandas (for data handling)

Matplotlib (for visualizations, optional)
