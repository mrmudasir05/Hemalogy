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
    os.makedirs(analyzer_dir_path, exist_ok=True)
    sliced_cells_dir = os.path.join(analyzer_dir_path, "Sliced Cells")
    os.makedirs(sliced_cells_dir, exist_ok=True)

    connected_cells_dir = os.path.join(analyzer_dir_path, "Connected Cells")
    os.makedirs(connected_cells_dir, exist_ok=True)

    boxes_dir = os.path.join(analyzer_dir_path, "Boxes")
    os.makedirs(boxes_dir, exist_ok=True)

    start_time = time.time()
    image = cv2.cvtColor(org_image, cv2.COLOR_BGR2GRAY)
    img_float = image.astype(np.float32)
    # Calculate mean intensity
    mean_intensity = np.mean(img_float)

    # Machine epsilon for numerical stability (avoid division by zero)
    eps = np.finfo(float).eps

    # Apply contrast stretch using vectorized operations (efficient)
    contrast1 = 1.0 / (1.0 + (mean_intensity / (img_float + eps)) ** 20)
    contrast1_uint8 = contrast1 * 255.0
    image = np.clip(contrast1_uint8, 0, 255).astype(np.uint8)
    # Apply a binary threshold to create a binary image
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height_min_range = int(image.shape[0] * 0.04)
    height_max_range = int(image.shape[0] * 0.09)
    width_min_range = int(image.shape[1] * 0.03)
    width_max_range = int(image.shape[1] * 0.08)
    # print(height_min_range,height_max_range)
    # print(width_min_range,width_max_range)

    roi_patches = []
    seperated_patches_bbox = {}

    for c, contour in enumerate(contours):
        # print(c)
        x, y, w, h = cv2.boundingRect(contour)
        roi_patches.append((x, y, w, h))

    connected_cell_patches = []

    for i, (x, y, w, h) in enumerate(roi_patches):
        roi_patch = org_image[y:y + h, x:x + w]

        if roi_patch.shape[0] in range(70, 150) and roi_patch.shape[1] in range(70, 160):
            path = os.path.join(sliced_cells_dir, f'roi_patch_{i}.png')

            seperated_patches_bbox[f'roi_patch_{i}.png'] = (x, y, w, h)
            cv2.imwrite(path, roi_patch)

        elif roi_patch.shape[0] >= 150 or roi_patch.shape[1] >= 160:
            path = os.path.join(connected_cells_dir, f'roi_patch_{i}.png')
            if roi_patch.any():
                cv2.imwrite(path, roi_patch)
                connected_cell_patches.append([x, y, x + w, y + h])

    from skimage.measure import label

    for n, patch in enumerate(connected_cell_patches):
        # print(patch)
        x0, y0, x1, y1 = patch
        sliced_image = image[y0:y1, x0:x1]
        org_sliced_image = org_image[y0:y1, x0:x1]
        th, im_th = cv2.threshold(sliced_image, 127, 255, cv2.THRESH_BINARY_INV);

        contours, _ = cv2.findContours(im_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(sliced_image.shape[:2], dtype=np.uint8)
        print(mask.shape)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
        mask = mask.astype(np.uint8)
        distance_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        # Normalize the distance transform for visualization (optional)
        distance = cv2.normalize(distance_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        coordinates = peak_local_max(distance, min_distance=15)

        for (x, y) in coordinates:
            # print(x,y)
            cv2.circle(distance, (int(y), int(x)), 5, (255, 255, 255), -1)  # Draw a red circle with radius 5

        markers = np.zeros_like(sliced_image, dtype=float)
        for x, i in enumerate(coordinates):
            markers[int(i[0]), int(i[1])] = x + 190
        if markers.any():
            probability_map = segmentation.random_walker(org_sliced_image[:, :, 1], markers, beta=10, mode='bf')

            # Assuming 'probability_map' is the output of your random walker algorithm
            labeled_regions = label(probability_map)

            # Iterate over each label to find the contours
            for region_label in np.unique(labeled_regions):
                # print("Region Label", region_label)
                if region_label == 0:
                    continue  # Skip the background

                # Create a binary mask for the current label
                probablity_mask = np.zeros(labeled_regions.shape, dtype="uint8")
                probablity_mask[labeled_regions == region_label] = 255
                im_out = cv2.bitwise_and(mask, probablity_mask)
                # Find contours
                contours, _ = cv2.findContours(im_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bounding_boxes = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    bounding_boxes.append((x, y, w, h))

                for patch_no, patch in enumerate(bounding_boxes):
                    x, y, w, h = patch
                    roi_patch = org_sliced_image[y:y + h, x:x + w]
                    # print(roi_patch)
                    if roi_patch.shape[0] in range(70, 180) and roi_patch.shape[1] in range(70, 195):
                        # if roi_patch.shape[0] >= 40 and roi_patch.shape[1] >=40:
                        path = os.path.join(sliced_cells_dir, f'roi_patch_{n}_{region_label}_{patch_no}.png')
                        cv2.imwrite(path, roi_patch)
                        x0_1 = x0 + x
                        y0_1 = y0 + y
                        x1_1 = w
                        y1_1 = h

                        seperated_patches_bbox[f'roi_patch_{n}_{region_label}_{patch_no}.png'] = (
                        x0_1, y0_1, x1_1, y1_1)



    end_time = time.time()
    total_time = end_time - start_time
    return total_time,sliced_cells_dir,seperated_patches_bbox




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
        # print(filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
        if image is None:
            print(f"Error reading image {image_path}")
            return None
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(grayscale_image, (128, 128))
        flattened_image = resized_image.flatten()
        flattened_image_test = np.array(flattened_image)
        image = pca.transform([flattened_image_test])
        if image is None:
            return None
        prediction = model.predict(image)
        predicted_class_index = int(np.max(prediction))
        predicted_class = class_names[predicted_class_index]
        if predicted_class is not None:
            class_counts[predicted_class] += 1

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



    return class_counts, [echinocytes_bbox, normal_bbox, others_bbox, schistocytes_bbox, tear_drop_cells_bbox]



