import argparse
import cv2
import logging
import sys
import time
from pathlib import Path
from draw_utils import draw_boxes
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path):
    """Load YOLOv8 model from local path."""
    try:
        logger.info(f"Loading model from {model_path}")
        model = YOLO(model_path)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)

def main(args):
    # Validate and resolve model path
    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Run download_model.py first.")
        return

    # Load model
    model = load_model(str(model_path))

    # Open video source
    try:
        if args.source.isdigit():
            cap = cv2.VideoCapture(int(args.source))
        else:
            cap = cv2.VideoCapture(args.source)
        
        if not cap.isOpened():
            logger.error(f"Could not open video source '{args.source}'.")
            return
        logger.info(f"Video source '{args.source}' opened successfully.")
    except Exception as e:
        logger.error(f"Error opening video source: {e}")
        return

    logger.info(f"Starting real-time object detection with YOLOv8 on source '{args.source}'. Press 'q' to quit.")

    prev_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Empty frame received from source. Ending detection.")
                break

            # Perform inference
            try:
                results = model(frame)
            except Exception as e:
                logger.error(f"Inference error: {e}")
                continue

            # Extract detections
            detections = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                scores = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                for box, score, cls in zip(boxes, scores, classes):
                    detections.append([*box, score, cls])

            # Draw boxes
            frame = draw_boxes(frame, detections, model.names, conf_threshold=args.conf_threshold)

            # Calculate and display FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show frame
            cv2.imshow('Real-Time Object Detection', frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Detection stopped by user.")
                break
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error during detection: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Resources released.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Object Detection with YOLOv8")
    parser.add_argument('--model_path', type=str, default='models/yolov8s.pt', help='Path to YOLOv8 model')
    parser.add_argument('--conf_threshold', type=float, default=0.25, help='Confidence threshold for detections')
    parser.add_argument('--source', type=str, default='0', help='Video source: camera index (e.g., 0) or video file/URL path')
    args = parser.parse_args()
    main(args)
