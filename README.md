# Real-Time Object Detection with YOLOv8

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete Python-based project for real-time object detection using a highly trained YOLOv8 model. This application captures video from a webcam, video file, or URL, performs inference on each frame, and displays bounding boxes with class labels and confidence scores. Designed for offline use, making it ideal for edge devices or environments without internet connectivity.

## Features

- **Real-Time Detection**: Processes video feed at high FPS for live object detection.
- **Highly Trained Model**: Utilizes YOLOv8s trained on the COCO dataset, capable of detecting 80+ object classes (e.g., person, car, dog).
- **Flexible Input Sources**: Supports webcam, video files, and IP camera URLs.
- **Customizable Thresholds**: Adjustable confidence threshold for filtering detections.
- **Error Handling**: Robust error handling for model loading, inference, and video capture.
- **Modular Code**: Clean, well-documented code with separate utilities for drawing and model loading.
- **Offline Operation**: Runs entirely locally after initial model download.
- **Resume-Worthy**: Demonstrates expertise in computer vision, deep learning, and Python development.

## Detectable Objects

The YOLOv8s model is trained on the COCO dataset and can detect the following 80 object classes:

- person
- bicycle
- car
- motorcycle
- airplane
- bus
- train
- truck
- boat
- traffic light
- fire hydrant
- stop sign
- parking meter
- bench
- bird
- cat
- dog
- horse
- sheep
- cow
- elephant
- bear
- zebra
- giraffe
- backpack
- umbrella
- handbag
- tie
- suitcase
- frisbee
- skis
- snowboard
- sports ball
- kite
- baseball bat
- baseball glove
- skateboard
- surfboard
- tennis racket
- bottle
- wine glass
- cup
- fork
- knife
- spoon
- bowl
- banana
- apple
- sandwich
- orange
- broccoli
- carrot
- hot dog
- pizza
- donut
- cake
- chair
- couch
- potted plant
- bed
- dining table
- toilet
- tv
- laptop
- mouse
- remote
- keyboard
- cell phone
- microwave
- oven
- toaster
- sink
- refrigerator
- book
- clock
- vase
- scissors
- teddy bear
- hair drier
- toothbrush

## Technologies Used

- **Python 3.8+**
- **Ultralytics YOLOv8**: Pre-trained object detection model with PyTorch backend.
- **OpenCV**: For video capture, image processing, and visualization.
- **NumPy**: For numerical operations.

## Project Structure

```
.
├── main.py                 # Main script for real-time detection
├── download_model.py       # Script to download YOLOv8 model weights
├── draw_utils.py           # Helper functions for drawing boxes
├── requirements.txt        # Python dependencies
├── models/                 # Directory for model weights (created by download_model.py)
├── .gitignore              # Git ignore file
├── LICENSE                 # MIT license
└── README.md               # Project documentation
```

## Setup Instructions

1. **Clone or Download the Project**:

   - Ensure you have Python 3.8+ installed.

2. **Install Dependencies**:

   ```
   pip install -r requirements.txt
   ```

3. **Download Model Weights**:

   ```
   python download_model.py
   ```

   This downloads the YOLOv8s model (~22MB) to the `models/` directory. Run this once with internet access.

4. **Run the Application**:
   ```
   python main.py
   ```
   - Opens webcam and starts detection.
   - Press 'q' to quit.

## Usage

- **Basic Run**: `python main.py`
- **Custom Confidence Threshold**: `python main.py --conf_threshold 0.5`
- **Custom Source**: `python main.py --source path/to/video.mp4` or `python main.py --source http://ip:port/video`
- **Custom Model Path**: `python main.py --model_path path/to/model.pt`

The application will display a window with the live video feed, overlaid with detected objects.

## Demo

- Run the script with a webcam connected.
- Point the camera at objects (e.g., a person or car).
- Observe bounding boxes, labels (e.g., "person: 0.89"), and FPS counter.

## Performance Notes

- Tested on CPU; GPU recommended for higher FPS.
- YOLOv8s offers improved accuracy and speed compared to YOLOv5.
- For better performance, consider larger YOLOv8 models or custom training on specific datasets.

## Future Improvements

- Implement object tracking across frames.
- Integrate with GUI frameworks like Tkinter for controls.
- Optimize for mobile/edge deployment.

## License

This project uses YOLOv8 under the AGPL-3.0 license. See Ultralytics for details.

## Contact

Built as a portfolio project. Feel free to fork and enhance!
