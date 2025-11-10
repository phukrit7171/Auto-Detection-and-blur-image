import cv2
import glob
import os
from ultralytics import YOLO

# --- CONFIGURATION ---

# 1. Path to your custom trained model
MODEL_PATH = "runs/detect/train/weights/best.pt"

# 2. Path to the FOLDER containing your test images
SOURCE_FOLDER = "./test/images"  # Or "path/to/your/test_images_folder"

# 3. Path to the FOLDER where blurred images will be saved
OUTPUT_FOLDER = "./blurred_images"

# 4. 'license_plate' is class 0 (the first in your .yaml names list)
CLASS_TO_BLUR = [0]

# 5. Blur intensity (must be odd numbers)
BLUR_KERNEL = (51, 51)
# ---------------------

# Load your custom YOLO model
print(f"Loading model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

# Create the output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Find all images (case-insensitive)
extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
image_paths = []
for ext in extensions:
    image_paths.extend(glob.glob(os.path.join(SOURCE_FOLDER, ext)))

print(f"Found {len(image_paths)} images to process...")

# Loop through each image
for img_path in image_paths:
    try:
        # Read the image
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Could not read {img_path}. Skipping.")
            continue
        
        # Run detection on the image
        results = model.predict(frame, classes=CLASS_TO_BLUR, verbose=False)

        # Loop through each detection result
        for res in results:
            if res.boxes is None or len(res.boxes) == 0:
                continue
                
            # Move boxes to CPU before converting to numpy
            boxes = res.boxes.cpu().numpy()  # Added .cpu() here!
            
            for box in boxes:
                # Get coordinates as integers
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                
                # Ensure coordinates are within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                # Ensure valid region (x2 > x1 and y2 > y1)
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Select the region of interest (ROI)
                roi = frame[y1:y2, x1:x2]
                
                # Apply Gaussian blur if ROI is not empty
                if roi.size > 0 and roi.shape[0] > 0 and roi.shape[1] > 0:
                    blurred_roi = cv2.GaussianBlur(roi, BLUR_KERNEL, 0)
                    # Put the blurred ROI back into the original frame
                    frame[y1:y2, x1:x2] = blurred_roi

        # Create the new file name and path
        basename = os.path.basename(img_path)
        output_path = os.path.join(OUTPUT_FOLDER, basename)

        # Save the blurred image
        cv2.imwrite(output_path, frame)
        print(f"✓ Processed: {basename}")

    except Exception as e:
        print(f"✗ Error processing {img_path}: {e}")

print(f"\n✓ Processing complete! All blurred images saved in: {OUTPUT_FOLDER}")