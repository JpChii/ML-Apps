# Data preprocessing
from typing import Dict, Tuple
import tensorflow as tf
import shutil
import os
import zipfile
from aslai.config import BASE_DIR, LOGS_DIR, DATA_DIR, logger
from pathlib import Path
import numpy as np
import string
import random
from tqdm import tqdm


# Function to unzip data
def unzip_data(data_dir: Path, zip_name: string):
    """
    Function to unzip a zipped dataset

    Args:
        data_dir (Path): Posix path where the data resides
        zip_name (String): String name of the zipped dataset

    Returns:
        Unzipped dataset in data_dir
    """

    # Unzipping the zipped artifact
    # Declaring the dataset name
    filename = str(data_dir) + zip_name
    # Initializing zipref instance
    zip_ref = zipfile.ZipFile(filename, "r")
    logger.info("Unzipping dataset to directory")
    # Unzipping dataset
    zip_ref.extractall(path=data_dir)
    zip_ref.close()

    # Removing __MACOSX if present in data directory
    if "__MACOSX" in os.listdir(data_dir):
        shutil.rmtree(str(data_dir) + "/__MACOSX/")

# Function to get percentage of images        
def get_percent_images(target_dir, new_dir, sample_amount=0.1, random_state=42):
  """
  Get  sample amount percentage of random images from target_dir and copy them to new_dir
  Args:
    target_dir (str) - file path of directory to extract images from
    new_dir (str) - new directory path to copy original images to
    sample_amount - Percentage of images to copy(eg 0.1 = 10%)
    random_state - random seed value
  """

  # Set random seed for reproducabality
  random.seed(random_state)

  # Get a list of dictionaries of images files in target_dir
  # eg -  [{"class_name":["2348348.jpg", "2829119.jpg"]}]
  images = [{dir_name: os.listdir(target_dir +  "/"+ dir_name)} for dir_name in os.listdir(target_dir)]
  for i in images:
    for k, v in i.items():
      # how many images to sample
      sample_number = round(int(len(v)*sample_amount))

      print(f"There are {len(v)} total images in '{target_dir+k}' so we're going to copay {sample_number} to the new directory.")
      print(f"Getting {sample_number} random images for {k}...")
      random_images = random.sample(v, sample_number)

      # Make new_dir for each key
      new_target_dir = new_dir + "/" + k
      print(f"Making dir: {new_target_dir}")
      os.makedirs(new_target_dir,exist_ok=True)

      # Keep track of images moved
      images_moved = []

      # Create file paths for original images and new file target
      print(f"Copying images from: {target_dir}\n\t \t to: {new_target_dir}")
      for filename in tqdm(random_images):
        og_path = target_dir + "/" + k + "/" + filename
        new_path = new_target_dir + "/" + filename

        # Copy images from OG path to new path
        shutil.copy2(og_path, new_path)
        images_moved.append(new_path)

        # Malke sure number of images moved is corect
      assert len(os.listdir(new_target_dir)) == sample_number
      assert len(images_moved) == sample_number
        
# Function to get label name
def get_label(filepath: str) -> str:
  """
  Function to get label name from filepath
  Args:
      filepath (str): filepath of the image
  Returns:
      label name
  """
  # for directory structure train -> class index -2 will be the class name
  return tf.strings.split(filepath, os.path.sep)[-2]

# Function to preprocess images from directory and return images and labels
def process_path(image_path: str, img_size: int = 224, label_mode: str = None) -> Tuple:
    """
    Function to preprocess the image from data directory
    Args:
        image_path (str): Path to the image to be preprocessed
        label_mode (str): Encoding type of label, allowed values - onehot, sparse
        IMG_SIZE (int): Target size for reshaping the image
    Returns:
        Tuple of tf.image and tf.label
    """

    logger.info(f"Using {label_mode} to encode the labels")
    logger.info("{img_size} image size is used for preprocessing")

    # Load the image and resie it to target IMG_SIZE
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [img_size, img_size])

    # Get the label name
    label_name = get_label(image_path)
    print(label_name)
    label = label_name == class_names

    # Encode the label
    label = tf.one_hot(tf.argmax(label), depth=29)

    return img, label
    

# Function to create tf.data.Dataset for model training
def create_dataset(data_dir: Path, train_dir: string, train_data_percent: float = 0.6, batch_size: int = 32, cleanup: bool = True) -> tf.data.Dataset:
    """
    Load the data from data directory and preprocess it to a fast loading `tf.data.Dataset`
    Args:
        data_dir (Path): Posix path where the data resides
        train_data_percent (float): Percentage of data to be used for training from the entire dataset
        batch_size (int): Batch size for the dataset
        cleanup (bool): Cleanup the directory after creating the dataset to free up space
    Returns:
        Datasets needed for model training and evaluation
    """

    logger.info(f"The data directory is: {data_dir}")
    logger.info(f"Train data percent used for this iteration is: {train_data_percent}")

    # Getting the image count from train and test directory
    TRAIN_DIR = Path(os.path.join(data_dir, train_dir))
    logger.info(f"Train directory is: {TRAIN_DIR}")
    print(f"Training dir: {TRAIN_DIR}")

    # Getting image count
    image_train_count = len(list(TRAIN_DIR.glob('*/*.jpg')))
    logger.info(f"Train images count: {image_train_count}")
    print(f"Image train count: {image_train_count}")

    # Getting class_names
    class_names = np.array(sorted([item.name for item in TRAIN_DIR.glob('*')]))
    logger.info(f"Class names in the dataset is: {class_names}")
    print(class_names)

    # Getting list of files
    train_ds = tf.data.Dataset.list_files(str(TRAIN_DIR) + '/*/*.jpg', shuffle=False)
    print(f"train_ds: {train_ds}")

    # Getting data based on train data percent
    train_size = int(image_train_count * train_data_percent)
    print(f"train_size: {train_size}")

    # logger.info(f"Using {train_size} images out of {image_train_count} training images")
    train_ds = train_ds.take(train_size)
    print(f"train_ds: {train_ds}")

    # Creating fast loading dataset
    logger.info("Beggining dataset creation")
    train_dataset = train_ds.map(map_func=process_path, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    # Clean up the train directory to free up space after creating the dataset
    if cleanup:
        logger.info(f"Deleting training directory: {TRAIN_DIR}")
        shutil.rmtree(TRAIN_DIR)

    return train_dataset
