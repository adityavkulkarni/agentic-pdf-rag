import os

import cv2
import pytesseract
import logging
import numpy as np
from pytesseract import Output

group_index = 1 #: Global counter for maintaining unique group IDs across multiple document pages
logger = logging.getLogger(__name__)


def draw_text_group_boxes(image_path,  output_path=None, groups_output_dir=None):
    global group_index
    # Read the image
    img = cv2.imread(image_path)
    original = img.copy()

    # Get detailed OCR data
    data = pytesseract.image_to_data(img, output_type=Output.DICT)

    # Extract valid text boxes with confidence > 30
    text_boxes = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 30 and data['text'][i].strip():
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            text_boxes.append([x, y, x + w, y + h])

    if not text_boxes:
        return img

    # Group nearby text boxes
    grouped_boxes = group_nearby_text_boxes(text_boxes)

    # Draw bounding boxes around each group
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
    cropped_images = []
    height, width = original.shape[:2]
    padding = 10
    groups = []
    for i, group_box in enumerate(grouped_boxes):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = group_box

        # Crop the group from original image
        x1 = max(x1 - padding, 0)
        x2 = min(x2 + padding, width)
        y1 = max(y1 - padding, 0)
        y2 = min(y2 + padding, height)
        cropped_group = original[y1:y2, x1:x2]

        # Save cropped image
        if groups_output_dir:
            crop_filename = f"group_{group_index:02d}.jpg"
            groups.append(crop_filename)
            crop_path = os.path.join(groups_output_dir, crop_filename)
            cv2.imwrite(crop_path, cropped_group)

            cropped_images.append({
                'filename': crop_filename,
                'path': crop_path,
                'bbox': (x1, y1, x2, y2),
                'original_bbox': (x1, y1, x2, y2)
            })

        # Draw bounding box on main image
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f'Group {group_index}', (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        group_index += 1
    if output_path:
        cv2.imwrite(output_path, img)

    return groups


def group_nearby_text_boxes(boxes, horizontal_threshold=50, vertical_threshold=30):
    if not boxes:
        return []

    # Convert to numpy array for easier manipulation
    boxes = np.array(boxes)
    groups = []
    used = set()

    for i, box in enumerate(boxes):
        if i in used:
            continue

        # Start a new group with current box
        current_group = [i]
        used.add(i)

        # Find all boxes that should be grouped with this one
        changed = True
        while changed:
            changed = False
            group_bounds = get_group_bounds([boxes[idx] for idx in current_group])

            for j, other_box in enumerate(boxes):
                if j in used:
                    continue

                if should_group_boxes(group_bounds, other_box, horizontal_threshold, vertical_threshold):
                    current_group.append(j)
                    used.add(j)
                    changed = True

        # Calculate final bounding box for the group
        group_boxes = [boxes[idx] for idx in current_group]
        final_bounds = get_group_bounds(group_boxes)
        groups.append(final_bounds)

    return groups


def get_group_bounds(boxes):
    if not boxes:
        return None

    boxes = np.array(boxes)
    x1 = np.min(boxes[:, 0])
    y1 = np.min(boxes[:, 1])
    x2 = np.max(boxes[:, 2])
    y2 = np.max(boxes[:, 3])

    return [x1, y1, x2, y2]


def should_group_boxes(group_bounds, box, h_threshold, v_threshold):
    gx1, gy1, gx2, gy2 = group_bounds
    bx1, by1, bx2, by2 = box

    # Check for overlap or proximity

    # Horizontal overlap or proximity
    h_overlap = not (bx2 < gx1 - h_threshold or bx1 > gx2 + h_threshold)

    # Vertical overlap or proximity
    v_overlap = not (by2 < gy1 - v_threshold or by1 > gy2 + v_threshold)

    # Group if there's overlap in both dimensions or if they're aligned and close
    if h_overlap and v_overlap:
        return True

    # Also group if they're vertically aligned and close horizontally
    v_aligned = not (by2 < gy1 or by1 > gy2)
    h_close = min(abs(bx1 - gx2), abs(gx1 - bx2)) < h_threshold * 2

    if v_aligned and h_close:
        return True

    # Group if they're horizontally aligned and close vertically
    h_aligned = not (bx2 < gx1 or bx1 > gx2)
    v_close = min(abs(by1 - gy2), abs(gy1 - by2)) < v_threshold * 2

    if h_aligned and v_close:
        return True

    return False


# Usage example
if __name__ == "__main__":
    image_path = "pdf_images/page_1.jpg"
    result1 = draw_text_group_boxes(image_path, "grouped_ocr.jpg")
    # Display results
    cv2.imshow('OCR-based Grouping', result1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


