from multiprocessing.spawn import import_main_path
from PIL import Image
from matplotlib.style import context
import tenseal #Used in preprocesser
from homo_eigenfaces_module import EigenfacesClient, EigenfacesServer
import numpy as np
import os
from tests import TestSuite

TRAINING_IMAGES_PATH = "Heinrich"
TEST_IMAGE_PATH = "test_images"

def load_images(image_root: str) -> np.array([]):
    images = [] #List for images
    image_directories = [] #List for image directories
    image_names = [] #List for image names
    image_labels = [] #List for image labels
    image_name_suffix = '.'

    #Store all image directories, if it does not start with a '.': 
    image_directories = [image for image in os.listdir(image_root) if not image.startswith(image_name_suffix)]
            
    #Within each directory, store all image names, if it does not start with a '.':
    for index, image_directory in enumerate(image_directories):
        in_directory_image_names = ([image_name for image_name in os.listdir(os.path.join(image_root, image_directory)) if not image_name.startswith(image_name_suffix)])
        image_names.append(in_directory_image_names)

    #Greyscale, rescale to default size, if necessary, and store in image list:
    for index, image_directory in enumerate(image_directories): 
        for image_name in image_names[index]:
            image = Image.open(os.path.join(image_root, image_directory, image_name)) #Open image
            images.append(image) #Append image to list of images
            image_labels.append(image_directory)
    #Return images and image names: 
    return images, image_labels

if __name__ == '__main__':
    # Create a homomorphic Eigenfaces client:
    Client = EigenfacesClient()
    Server = EigenfacesServer(Client._n_components_comparison, Client._distance_comparison, 
    Client._goldschmidt_initializer, Client._reencrypt_mat,Client._reencrypt_vec,Client._decrypt_vec)

    # Load training images and test image from paths: 
    training_images, training_image_labels = load_images(TRAINING_IMAGES_PATH)
    test_images, test_image_labels = load_images(TEST_IMAGE_PATH)

    # Preprocess the images, using the client:
    normalized_training_images = Client.Image_preprocesser(training_images)
    # Represent the training images as vectors, using the client:
    vectorized_training_images = Client.Image_vector_representation(normalized_training_images)
    
    # Preprocess the images, using the client:
    normalized_test_images = Client.Image_preprocesser(test_images)
    # Encrypt images: 
    encrypted_normalized_training_images = np.empty(shape=0)
    for i in range(0,len(normalized_training_images)):
        encrypted_normalized_training_images = np.append(encrypted_normalized_training_images,Client.Encrypt(normalized_training_images[i]))
    encrypted_vectorized_training_images = Client.Encrypt(vectorized_training_images)
    #print("vec = ",Client.Decrypt(encrypted_vectorized_training_images))
    #encrypted_normalized_test_images = Client.Encrypt(normalized_test_images)
    
    #Testing the module: 
    tests = TestSuite(Server)
    tests.computation_time_training(encrypted_normalized_training_images, encrypted_vectorized_training_images)
    #tests.computation_time_classification(normalized_test_images, training_image_labels)
    #tests.prediction_accuracy(test_image_labels)