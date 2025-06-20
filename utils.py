import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage import filters, morphology, segmentation, measure, feature
from skimage.measure import label
import os
import time
import pickle
import shutil

# Load the machine learning model
with open(r"svm_model_version_3.0.sav", 'rb') as model_file:
    model = pickle.load(model_file)

# Assuming PCA is also saved and needs to be loaded
with open(r"pca_version_3.0.pkl", 'rb') as pca_file:
    pca = pickle.load(pca_file)


def delete_directory_contents(dir_path):

    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except OSError as e:
            print(f"Error deleting {file_path}: {e}")


def slice_individual_cells(org_image, analyzer_dir_path):
    """
    Advanced cell detection and segmentation function with multiple complementary approaches
    to maximize the number of detected blood cells in an image.
    
    Args:
        org_image: Original blood smear image (BGR format)
        analyzer_dir_path: Directory to save analysis results
        
    Returns:
        total_time: Processing time
        sliced_cells_dir: Directory containing extracted cell images
        seperated_patches_bbox: Dictionary of bounding boxes for each cell
    """
    # Create necessary directories
    os.makedirs(analyzer_dir_path, exist_ok=True)
    sliced_cells_dir = os.path.join(analyzer_dir_path, "Sliced Cells")
    os.makedirs(sliced_cells_dir, exist_ok=True)
    connected_cells_dir = os.path.join(analyzer_dir_path, "Connected Cells")
    os.makedirs(connected_cells_dir, exist_ok=True)
    boxes_dir = os.path.join(analyzer_dir_path, "Boxes")
    os.makedirs(boxes_dir, exist_ok=True)

    start_time = time.time()
    seperated_patches_bbox = {}
    
    # Create a copy of the original image for visualization
    org_vis = org_image.copy()
    
    # Convert to grayscale
    if len(org_image.shape) == 3:
        gray = cv2.cvtColor(org_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = org_image.copy()
    
    # PHASE 1: Pre-processing with multiple techniques
    # ------------------------------------------------
    
    # 1. CLAHE enhancement (brings out details)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe_enhanced = clahe.apply(gray)
    
    # 2. Contrast enhancement using exponential transformation
    img_float = gray.astype(np.float32)
    mean_intensity = np.mean(img_float)
    eps = np.finfo(float).eps
    # Try different power values for contrast enhancement
    contrast_enhanced_images = []
    for power in [5, 10, 15]:  # Multiple power values for different contrast levels
        contrast = 1.0 / (1.0 + (mean_intensity / (img_float + eps)) ** power)
        contrast_uint8 = contrast * 255.0
        contrast_img = np.clip(contrast_uint8, 0, 255).astype(np.uint8)
        contrast_enhanced_images.append(contrast_img)
    
    # 3. Gaussian blur to reduce noise (helps with thresholding)
    blurred = cv2.GaussianBlur(clahe_enhanced, (5, 5), 0)
    
    # PHASE 2: Multi-level thresholding
    # ---------------------------------
    binary_images = []
    
    # Apply multiple thresholding methods to each pre-processed image
    preprocessed_images = [gray, clahe_enhanced, blurred] + contrast_enhanced_images
    
    for img in preprocessed_images:
        # Otsu thresholding
        _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary_images.append(otsu)
        
        # Adaptive thresholding - smaller block size for small cells
        adaptive1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)
        binary_images.append(adaptive1)
        
        # Adaptive thresholding - larger block size for larger cells
        adaptive2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 21, 3)
        binary_images.append(adaptive2)
    
    # Combine all binary images
    combined_binary = np.zeros_like(gray)
    for binary in binary_images:
        combined_binary = cv2.bitwise_or(combined_binary, binary)
    
    # PHASE 3: Morphological operations to clean up binary image
    # ---------------------------------------------------------
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_medium = np.ones((5, 5), np.uint8)
    
    # Remove small noise
    opening = cv2.morphologyEx(combined_binary, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # Close small holes within cells
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
    
    # Save pre-processed image for debugging
    safe_imwrite(os.path.join(connected_cells_dir, 'preprocessed_binary.png'), closing)
    
    # PHASE 4: Advanced contour detection
    # ----------------------------------
    
    # Find contours on cleaned binary image
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw all contours for visualization
    contour_img = org_image.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    safe_imwrite(os.path.join(connected_cells_dir, 'all_contours.png'), contour_img)
    
    # PHASE 5: Contour analysis and cell extraction
    # --------------------------------------------
    
    # Calculate size ranges based on image dimensions
    img_area = gray.shape[0] * gray.shape[1]
    img_diagonal = np.sqrt(gray.shape[0]**2 + gray.shape[1]**2)
    
    # Estimate cell size based on image resolution
    # For a typical blood smear at 40x magnification, RBCs are about 1-2% of the image diagonal
    min_cell_dim = max(int(img_diagonal * 0.01), 25)  # At least 25 pixels
    max_cell_dim = int(img_diagonal * 0.08)  # Up to 8% of diagonal
    
    # Area thresholds
    min_cell_area = int(min_cell_dim**2 * 0.5)  # Allow for non-square cells
    max_cell_area = int(max_cell_dim**2 * 1.5)  # Allow for larger irregular cells
    
    # Maximum size for connected cells
    max_connected_dim = int(img_diagonal * 0.25)
    
    # Lists to store individual cells and connected cell regions
    individual_cells = []
    connected_regions = []
    cell_count = 0
    
    # Process each contour
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        # Skip very small contours (noise)
        if area < min_cell_area:
            continue
            
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate aspect ratio and circularity (useful for cell identification)
        aspect_ratio = float(w) / h if h > 0 else 0
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Check if this might be a single cell
        is_single_cell = (
            min_cell_dim <= w <= max_cell_dim and 
            min_cell_dim <= h <= max_cell_dim and
            0.5 <= aspect_ratio <= 2.0 and  # Not too elongated
            circularity > 0.4  # Somewhat circular
        )
        
        # Check if this might be a cluster of cells
        is_cell_cluster = (
            (w > max_cell_dim or h > max_cell_dim) and 
            w <= max_connected_dim and h <= max_connected_dim and
            area > max_cell_area
        )
        
        # Extract region from original image
        roi = org_image[y:y+h, x:x+w]
        
        if is_single_cell:
            # Process single cell
            cell_filename = f'cell_{cell_count}.png'
            cell_path = os.path.join(sliced_cells_dir, cell_filename)
            
            # Add some padding around the cell
            padding = max(5, int(min(w, h) * 0.1))
            
            # Ensure padded coordinates are within image bounds
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(org_image.shape[1], x + w + padding)
            y_end = min(org_image.shape[0], y + h + padding)
            
            # Extract padded cell
            padded_cell = org_image[y_start:y_end, x_start:x_end]
            
            # Save the cell
            if padded_cell.size > 0:
                safe_imwrite(cell_path, padded_cell)
                
                # Store bounding box information
                seperated_patches_bbox[cell_filename] = (x, y, w, h)
                
                # Draw rectangle on visualization image (green for single cells)
                cv2.rectangle(org_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                cell_count += 1
            
        elif is_cell_cluster:
            # This region likely contains multiple cells
            connected_regions.append((x, y, w, h))
            
            # Save region for debugging
            cluster_path = os.path.join(connected_cells_dir, f'cluster_{i}.png')
            if roi.size > 0:
                safe_imwrite(cluster_path, roi)
    
    # PHASE 6: Process cell clusters with advanced watershed segmentation
    # -----------------------------------------------------------------
    cluster_cell_count = 0
    
    for i, (x0, y0, w, h) in enumerate(connected_regions):
        # Extract the region containing multiple cells
        cluster_img = org_image[y0:y0+h, x0:x0+w]
        
        # Convert to grayscale if needed
        if len(cluster_img.shape) == 3:
            cluster_gray = cv2.cvtColor(cluster_img, cv2.COLOR_BGR2GRAY)
        else:
            cluster_gray = cluster_img.copy()
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(cluster_gray)
        
        # Otsu thresholding
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        
        # Noise removal and morphological operations
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Find sure foreground area using distance transform
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        
        # Dynamic thresholding based on the distance transform distribution
        dist_max = dist_transform.max()
        if dist_max > 0:
            # Use a lower threshold to detect more cells in clusters
            _, sure_fg = cv2.threshold(dist_transform, 0.3*dist_max, 255, 0)
            
            # Improved marker generation for watershed
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            
            # Apply connected components to get markers
            _, markers = cv2.connectedComponents(sure_fg)
            
            # Add one to all labels so that background is 1 instead of 0
            markers = markers + 1
            
            # Mark the unknown region with 0
            markers[unknown == 255] = 0
            
            # Apply watershed
            markers_copy = markers.copy()
            watershed_markers = cv2.watershed(cluster_img, markers_copy)
            
            # Process each cell found by watershed
            for label in np.unique(watershed_markers):
                if label <= 1:  # Skip background and boundary
                    continue
                    
                # Create mask for this cell
                cell_mask = np.zeros_like(cluster_gray, dtype=np.uint8)
                cell_mask[watershed_markers == label] = 255
                
                # Find contours of the cell
                cell_contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(cell_contours) == 0:
                    continue
                    
                # Get the largest contour
                largest_contour = max(cell_contours, key=cv2.contourArea)
                
                # Calculate area and check if it's large enough
                area = cv2.contourArea(largest_contour)
                if area < min_cell_area:
                    continue
                    
                # Calculate circularity (helps filter out non-cells)
                perimeter = cv2.arcLength(largest_contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                if circularity < 0.3:  # Too irregular to be a cell
                    continue
                    
                # Get bounding box
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Check dimensions
                if w < min_cell_dim or h < min_cell_dim:
                    continue
                    
                # Extract the cell
                cell_img = cluster_img[y:y+h, x:x+w]
                
                # Check aspect ratio
                aspect_ratio = float(w) / h if h > 0 else 0
                if aspect_ratio < 0.4 or aspect_ratio > 2.5:  # Too elongated
                    continue
                
                # Save the cell
                cell_filename = f'cluster_{i}_cell_{cluster_cell_count}.png'
                cell_path = os.path.join(sliced_cells_dir, cell_filename)
                
                if cell_img.size > 0:
                    safe_imwrite(cell_path, cell_img)
                    
                    # Calculate global coordinates
                    x_global = x0 + x
                    y_global = y0 + y
                    
                    # Store bounding box
                    seperated_patches_bbox[cell_filename] = (x_global, y_global, w, h)
                    
                    # Draw rectangle on visualization (red for cluster cells)
                    cv2.rectangle(org_vis, (x_global, y_global), (x_global+w, y_global+h), (0, 0, 255), 2)
                    
                    cluster_cell_count += 1
    
    # PHASE 7: Try a different approach - detect cells using circle detection
    # ---------------------------------------------------------------------
    
    # Apply HoughCircles to detect circular cells
    circles = None
    try:
        # Try multiple parameters for circle detection
        for dp in [1, 1.5]:
            for minDist in [20, 30, 40]:
                for param1 in [50, 100]:
                    for param2 in [25, 30, 35]:
                        circles = cv2.HoughCircles(
                            clahe_enhanced, 
                            cv2.HOUGH_GRADIENT, 
                            dp=dp, 
                            minDist=minDist,
                            param1=param1, 
                            param2=param2, 
                            minRadius=int(min_cell_dim/2*0.8), 
                            maxRadius=int(max_cell_dim/2*1.2)
                        )
                        
                        if circles is not None and len(circles) > 0:
                            break
                    if circles is not None and len(circles) > 0:
                        break
                if circles is not None and len(circles) > 0:
                    break
            if circles is not None and len(circles) > 0:
                break
    except Exception as e:
        print(f"Circle detection failed: {e}")
        
    # Process detected circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for i, (x, y, r) in enumerate(circles[0, :]):
            # Create unique filename
            cell_filename = f'circle_cell_{i}.png'
            
            # Define box around circle
            x1 = max(0, x - r)
            y1 = max(0, y - r)
            x2 = min(org_image.shape[1], x + r)
            y2 = min(org_image.shape[0], y + r)
            
            if x2 - x1 < min_cell_dim or y2 - y1 < min_cell_dim:
                continue
                
            # Extract cell region
            cell_img = org_image[y1:y2, x1:x2]
            
            # Check if we already have this cell (avoid duplicates)
            is_duplicate = False
            for (ex, ey, ew, eh) in seperated_patches_bbox.values():
                # Check if center of this circle is inside an existing bounding box
                if (ex <= x <= ex+ew) and (ey <= y <= ey+eh):
                    is_duplicate = True
                    break
                    
            if not is_duplicate and cell_img.size > 0:
                # Save cell
                cell_path = os.path.join(sliced_cells_dir, cell_filename)
                safe_imwrite(cell_path, cell_img)
                
                # Store bounding box
                seperated_patches_bbox[cell_filename] = (x1, y1, x2-x1, y2-y1)
                
                # Draw circle on visualization (blue for circle-detected cells)
                cv2.circle(org_vis, (x, y), r, (255, 0, 0), 2)
    
    # Save visualization with all detected cells
    safe_imwrite(os.path.join(boxes_dir, 'all_cells_detected.png'), org_vis)
    
    # Report the final cell count
    total_cells = len(seperated_patches_bbox)
    print(f"Found {total_cells} cells in total")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return total_time, sliced_cells_dir, seperated_patches_bbox




# Function to process all images in a folder and count the classes
def predict_images(folder_path, seperated_patches_bbox):
    class_names = ["Echinocytes", "Normal-RBCs", "others", "Schistocytes", "Tear_drop_cells"]

    # Initialize a dictionary to count occurrences of each class
    class_counts = {class_name: 0 for class_name in class_names}


    echinocytes_bbox = {}
    normal_bbox = {}
    others_bbox = {}
    schistocytes_bbox = {}
    tear_drop_cells_bbox = {}


    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            
            if image is None or image.size == 0:
                print(f"Error reading image {image_path} or image is empty")
                continue
                
            try:
                grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                resized_image = cv2.resize(grayscale_image, (128, 128))
                flattened_image = resized_image.flatten()
                flattened_image_test = np.array(flattened_image)
                image_features = pca.transform([flattened_image_test])
                
                if image_features is None:
                    print(f"Error: PCA transformation failed for {filename}")
                    continue
                    
                prediction = model.predict(image_features)
                predicted_class_index = int(np.max(prediction))
                predicted_class = class_names[predicted_class_index]
                
                if predicted_class is not None:
                    class_counts[predicted_class] += 1

                # Make sure the filename exists in the bbox dictionary
                if filename not in seperated_patches_bbox:
                    print(f"Warning: {filename} not found in bounding box dictionary")
                    continue
                    
                bbox_in_org_image = seperated_patches_bbox[filename]
                
                if predicted_class == 'Echinocytes':
                    echinocytes_bbox[filename] = bbox_in_org_image
                elif predicted_class == 'Normal-RBCs':
                    normal_bbox[filename] = bbox_in_org_image
                elif predicted_class == 'others':
                    others_bbox[filename] = bbox_in_org_image
                elif predicted_class == 'Schistocytes':
                    schistocytes_bbox[filename] = bbox_in_org_image
                elif predicted_class == 'Tear_drop_cells':
                    tear_drop_cells_bbox[filename] = bbox_in_org_image
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue



    return class_counts, [echinocytes_bbox, normal_bbox, others_bbox, schistocytes_bbox, tear_drop_cells_bbox]


def watershed_segmentation(img):
    """Separate touching cells using watershed algorithm"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # Threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # Noise removal
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all labels so that background is 1 instead of 0
    markers = markers + 1
    
    # Mark the unknown region with 0
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv2.watershed(img, markers)
    
    return markers

def is_valid_cell(cell_img, min_size=30, circularity_threshold=0.5):
    """Validate if a segmented region is likely to be a blood cell"""
    if cell_img.size == 0 or cell_img.shape[0] < min_size or cell_img.shape[1] < min_size:
        return False
        
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY) if len(cell_img.shape) == 3 else cell_img
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # Find contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
        
    # Calculate circularity for the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    if perimeter == 0:
        return False
        
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    return circularity > circularity_threshold

def safe_imwrite(filepath, img):
    """
    Safely write an image to disk with proper error checking.
    Returns True if successful, False otherwise.
    """
    if img is None or img.size == 0:
        print(f"Warning: Attempted to write empty image to {filepath}")
        return False
    
    try:
        cv2.imwrite(filepath, img)
        return True
    except Exception as e:
        print(f"Error saving image to {filepath}: {e}")
        return False
