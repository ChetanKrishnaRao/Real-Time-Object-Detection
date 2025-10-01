import cv2

def draw_boxes(img, results, classes, colors=None, conf_threshold=0.25):
    """
    Draw bounding boxes and labels on the image.

    Args:
        img (np.array): Image to draw on.
        results (list): List of detections with [x1, y1, x2, y2, conf, cls].
        classes (list): List of class names.
        colors (dict): Optional dict of colors for each class.
        conf_threshold (float): Confidence threshold to filter boxes.

    Returns:
        img (np.array): Image with boxes drawn.
    """
    if colors is None:
        colors = {}

    for *box, conf, cls in results:
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, box)
        label = f"{classes[int(cls)]}: {conf:.2f}"
        color = colors.get(int(cls), (0, 255, 0))

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)

        # Put label text
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return img
