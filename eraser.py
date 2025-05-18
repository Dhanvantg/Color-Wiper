import PIL
from PIL import Image
import PIL.ImageEnhance
import pdf2image
import numpy as np
import cv2
import os

def convert_pdf_to_images(pdf_path):
    """Convert PDF to a list of PIL Images."""
    # Get the absolute path to the poppler bin directory
    poppler_path = os.path.abspath(os.path.join("poppler", "poppler-23.11.0", "Library", "bin"))
    return pdf2image.convert_from_path(pdf_path, poppler_path=poppler_path)

def detect_red_regions(image):
    """Detect red regions in an image using HSV color ranges."""
    # Convert PIL Image to OpenCV format
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define ranges for red color in HSV
    # Red wraps around in HSV, so we need two ranges
    lower_red1 = np.array([0, 10, 10], dtype=np.uint8)    # Lower range for red
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)  # Upper range for red
    lower_red2 = np.array([150, 10, 10], dtype=np.uint8)   # Lower range for red (wrapped)
    upper_red2 = np.array([230, 255, 255], dtype=np.uint8) # Upper range for red (wrapped)
    
    # Create masks for both red ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Combine the masks
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Clean up the mask
    kernel = np.ones((1,1), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def find_similar_patterns(image, initial_mask):
    """Find additional red regions based on pattern similarity and local color analysis."""
    # Convert to HSV for color analysis
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Find contours in the initial mask
    contours, _ = cv2.findContours(initial_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a new mask for additional patterns
    pattern_mask = np.zeros_like(initial_mask)
    
    # Process each contour
    for contour in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Skip very small contours
        if w < 5 or h < 5:
            continue
        
        # Get the color statistics of the current contour region
        roi_hsv = hsv[y:y+h, x:x+w]
        roi_mask = initial_mask[y:y+h, x:x+w]
        mean_color = cv2.mean(roi_hsv, mask=roi_mask)
        
        # Find similar contours in the image
        for other_contour in contours:
            # Skip if it's the same contour
            if np.array_equal(contour, other_contour):
                continue
            
            # Calculate similarity between contours
            similarity = cv2.matchShapes(contour, other_contour, cv2.CONTOURS_MATCH_I2, 0)
            
            # If contours are similar enough
            if similarity < 0.5:  # Strict threshold for pattern matching
                # Get the region around the similar contour
                ox, oy, ow, oh = cv2.boundingRect(other_contour)
                
                # Skip if this region is already marked
                if np.sum(initial_mask[oy:oy+oh, ox:ox+ow]) > 0:
                    continue
                
                # Analyze color in the similar region
                similar_roi_hsv = hsv[oy:oy+oh, ox:ox+ow]
                
                # Check if the region has any red hue
                # Use a wider range for hue to catch potential red marks
                hue_range = 1000  # Allow some variation in hue
                sat_threshold = 0  # Minimum saturation to consider
                val_threshold = 0  # Minimum value to consider
                
                # Create a mask for potential red pixels in the region
                red_mask = np.zeros((oh, ow), dtype=np.uint8)
                
                # Check both red ranges (0-10 and 170-180)
                lower_red1 = np.array([max(0, int(mean_color[0] - hue_range)), sat_threshold, val_threshold])
                upper_red1 = np.array([min(10, int(mean_color[0] + hue_range)), 255, 255])
                lower_red2 = np.array([max(170, int(mean_color[0] - hue_range)), sat_threshold, val_threshold])
                upper_red2 = np.array([min(180, int(mean_color[0] + hue_range)), 255, 255])
                
                mask1 = cv2.inRange(similar_roi_hsv, lower_red1, upper_red1)
                mask2 = cv2.inRange(similar_roi_hsv, lower_red2, upper_red2)
                red_mask = cv2.bitwise_or(mask1, mask2)
                
                # Only add to pattern mask if there's significant red content
                if np.sum(red_mask) > (ow * oh * 0.0000001):  # At least 5% of the region should be red
                    cv2.drawContours(pattern_mask, [other_contour], -1, 255, -1)
    
    # Combine with initial mask
    final_mask = cv2.bitwise_or(initial_mask, pattern_mask)
    
    # Final cleanup with minimal kernel to preserve details
    kernel = np.ones((1,1), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    
    return final_mask

def process_image(image, mask):
    """Process the image by removing red regions."""
    # Convert PIL Image to OpenCV format
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Inpaint the red regions
    processed = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    
    return Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))

def visualize_red_regions(image):
    """Create a visualization of detected red regions on a white background."""
    # Convert PIL Image to OpenCV format
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # First detect red regions using color
    initial_mask = detect_red_regions(image)
    
    # Then find similar patterns
    final_mask = find_similar_patterns(image, initial_mask)
    
    # Create a white background
    height, width = img.shape[:2]
    white_bg = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Create visualization
    # Red regions will be shown in red, rest in white
    visualization = white_bg.copy()
    visualization[final_mask > 0] = [0, 0, 255]  # Red in BGR
    
    return Image.fromarray(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)), final_mask

def process_pdf(pdf_path, output_dir):
    """Process a PDF file and save both annotations and processed images."""
    # Create output directories if they don't exist
    annotations_dir = os.path.join(output_dir, "annotations")
    processed_dir = os.path.join(output_dir, "processed")
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Convert PDF to images
    print(f"Converting PDF: {pdf_path}")
    images = convert_pdf_to_images(pdf_path)
    total_pages = len(images)
    print(f"Total pages found: {total_pages}")
    
    # Process each page
    for i, image in enumerate(images, 1):
        print(f"Processing page {i}/{total_pages}")

        # Create visualization of red regions and get mask
        visualization, mask = visualize_red_regions(image)
        
        # Save annotation
        annotation_path = os.path.join(annotations_dir, f"red_regions_page_{i}.png")
        visualization.save(annotation_path)
        print(f"Saved annotation to: {annotation_path}")
        
        # If red regions were detected, process and save the image
        if mask is not None:
            processed_image = process_image(image, mask)
            processed_path = os.path.join(processed_dir, f"processed_page_{i}.png")
            processed_image.save(processed_path)
            print(f"Saved processed image to: {processed_path}")

if __name__ == "__main__":
    # Example usage
    pdf_path = "cp_midsem.pdf"  # Replace with your PDF path
    output_dir = "output"
    process_pdf(pdf_path, output_dir)
