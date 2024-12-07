import os
from config import *
from web import *

if __name__ == '__main__':
    # Define target paths for training, testing, and validation datasets
    target_paths = {
        'training_image': os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'imagesTr'),
        'training_label': os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'labelsTr'),
        'testing_image': os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'imagesTs'),
        'testing_label': os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'labelsTs'),
        'validation_image': os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'imagesVal'),
        'validation_label': os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'labelsVal'),
    }

    try:
        # Convert all .jpg files to .png format in the dataset directories
        for path_each in target_paths.values():
            convert_jpg_to_png_all_from_path(path_each)

        # Find all unique labels in the training label data
        unique_values = find_unique_labels(target_paths['training_label'])

        # Convert the labels in training, validation, and testing datasets
        # to match the unique values found in the training labels
        convert_label_by_searchsorted(target_paths['training_label'], unique_values)
        convert_label_by_searchsorted(target_paths['validation_label'], unique_values)
        convert_label_by_searchsorted(target_paths['testing_label'], unique_values)

        # Generate the dataset.json file for the current dataset
        print_web(f"Generating dataset.json for {os.environ['current_dataset']}")

        # Get the first image in the training image directory to determine its properties
        npimg_path = os.path.join(target_paths['training_image'], os.listdir(target_paths['training_image'])[0])
        npimg = cv2.imread(npimg_path, cv2.IMREAD_UNCHANGED)

        # Determine the number of image channels (3 for RGB, 1 for grayscale)
        img_channel = 3 if len(npimg.shape) == 3 else 1

        # Count the number of unique label classes
        label_class_num = len(unique_values)

        # Generate a unique dataset ID
        dataset_id = len(os.listdir(os.environ['medseg_raw'])) + 1
        dataset_id = "{:03}".format(dataset_id)

        # Determine the size of the image (assuming square images)
        image_size = npimg.shape[0]

        # Save the dataset metadata to a dataset.json file
        save_dataset_json(dataset_id, os.environ['current_dataset'], image_size, img_channel, label_class_num)

    except Exception as e:
        # Handle and log any errors that occur during processing
        print_web(f"Error: {e}")
        raise e
