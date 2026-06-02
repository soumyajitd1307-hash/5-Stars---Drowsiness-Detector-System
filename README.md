Abstract
The Smart Drowsiness Detection System is a computer vision–based safety application designed to monitor a driver's eye activity in real time and detect signs of drowsiness. The system uses a webcam and MediaPipe Face Mesh to track facial landmarks and calculate the Eye Aspect Ratio (EAR). An automatic calibration process determines a personalized EAR threshold for each user, improving detection accuracy. When the driver's eyes remain closed for a prolonged period, the system generates an alert through an Arduino-controlled buzzer and LED, helping prevent accidents caused by fatigue.

Introduction
Driver drowsiness is one of the major causes of road accidents worldwide. Long driving hours, lack of sleep, and fatigue can significantly reduce a driver's reaction time and concentration. To address this problem, a real-time drowsiness detection system has been developed using Python, OpenCV, MediaPipe, and Arduino. The system continuously monitors the driver's eye movements through a webcam and provides immediate alerts when signs of drowsiness are detected. Unlike traditional systems that require manual threshold adjustment, this project includes an automatic calibration feature that adapts to individual users.

Aim
The main aim of this project is to develop a real-time driver monitoring system capable of:

Detecting drowsiness by analyzing eye closure duration.

Automatically calibrating detection parameters according to the user's facial features.

Providing instant audio and visual alerts using a buzzer and LED.

Improving road safety by reducing accidents caused by driver fatigue.

Working Principle
1. Video Acquisition
A webcam continuously captures live video frames of the driver's face.

2. Face and Eye Landmark Detection
The captured frames are processed using MediaPipe Face Mesh, which identifies facial landmarks, including the eye regions.

3. Automatic EAR Calibration
At startup, the system asks the user to keep their eyes open for a few seconds. During this period:

Multiple EAR samples are collected.

The average open-eye EAR is calculated.

A personalized EAR threshold is generated automatically.

4. Eye Aspect Ratio (EAR) Calculation
The Eye Aspect Ratio is computed using selected eye landmarks. The EAR value decreases when the eyes close and increases when they are open.

5. Drowsiness Monitoring
The system continuously compares the current EAR value with the calibrated threshold:

If EAR falls below the threshold, the eyes are considered closed.

The duration of eye closure is measured.

6. Warning Stage
If the eyes remain closed for approximately 3 seconds, a warning message is displayed on the screen.

7. Alert Stage
If the eyes remain closed for approximately 7 seconds:

A drowsiness event is detected.

The Python program sends a serial signal ('1') to the Arduino.

8. Alert Generation
Upon receiving the signal:

The Arduino activates the buzzer.

The LED turns ON.

The buzzer continues beeping until the driver's eyes reopen.

9. Alert Reset
When the driver's eyes open again:

Python sends a stop signal ('0').

The Arduino turns OFF the buzzer and LED.

Monitoring resumes normally.

Materials Required
Hardware
Laptop/PC

USB Webcam (or built-in webcam)

Arduino Uno

Active Buzzer (5V)

LED

220 Ω Resistor

Jumper Wires

USB Cable for Arduino

Software
Python 3.x

Arduino IDE

OpenCV

MediaPipe

NumPy

PySerial

Arduino Connections
Component	Arduino Pin
Buzzer (+)	Digital Pin 8
Buzzer (-)	GND
LED (+)	Pin 13 through 220 Ω resistor
LED (-)	GND
Conclusion
The Smart Drowsiness Detection System successfully detects driver fatigue in real time using computer vision techniques. By utilizing MediaPipe Face Mesh and automatic EAR calibration, the system provides accurate and personalized drowsiness detection without manual threshold tuning. The integration of Arduino with a buzzer and LED ensures immediate alerts whenever prolonged eye closure is detected. The project demonstrates an effective, low-cost, and practical solution for enhancing driver safety and reducing accidents caused by drowsiness.
