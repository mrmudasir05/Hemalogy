import pickle
import pandas as pd
import base64
import io
from PIL import Image
import cv2
from utils import predict_images, slice_individual_cells, delete_directory_contents
import os

import tempfile

def main(img_path):
    # Create a temporary directory for analysis data
    with tempfile.TemporaryDirectory() as analyzer_dir_path:
        print("Temporary Directory for Analysis Data:", analyzer_dir_path)

        # Initialize p_img in case org_image is not found
        p_img = None

        # Read the image using OpenCV
        org_image = cv2.imread(img_path)
        if org_image is None:
            print(f"Error: Image file not found at {img_path}")
            return None, None
        else:
            # Slice individual cells
            total_time, sliced_cells_dir, separated_patches_bbox = slice_individual_cells(org_image, analyzer_dir_path)

            # Predict images
            a, bbox = predict_images(sliced_cells_dir, separated_patches_bbox)
            
            # Define colors for the bounding boxes
            colors = [(0, 0, 255), (0, 0, 0), (255, 255, 255), (0, 255, 0), (255, 0, 0)]
            
            # Draw bounding boxes on the original image
            for i in range(5):
                for k in list(bbox[i].values()):
                    x, y, w, h = k
                    p_img = cv2.rectangle(org_image, (x, y), (x + w, y + h), colors[i], 2)

            # Clean up: delete the temp directory contents
            delete_directory_contents(analyzer_dir_path)

    return p_img, a