import logging
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# URL for YOLOv8 small model weights
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
MODEL_PATH = Path("models/yolov8s.pt")

def download_model():
    """Download YOLOv8 model weights if not present."""
    try:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        if MODEL_PATH.exists():
            logger.info("Model already downloaded.")
            return

        logger.info("Downloading YOLOv8 model...")
        response = requests.get(MODEL_URL, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info("Model downloaded successfully.")
    except requests.RequestException as e:
        logger.error(f"Failed to download model: {e}")
        raise
    except OSError as e:
        logger.error(f"Failed to save model file: {e}")
        raise

if __name__ == "__main__":
    download_model()
