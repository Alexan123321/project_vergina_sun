from eigenfaces_module import Eigenfaces

TRAINING_IMAGES_PATH = "training_path"
TEST_IMAGE_PATH = "test_path"

if __name__ == '__main__':
    # Create a homomorphic Eigenfaces model:
    eigenfaces_model = Eigenfaces()

    print("Starting Eigenfaces module...")

    # Train the model using the path of the training images:
    eigenfaces_model.train(TRAINING_IMAGES_PATH)

    # Classify a test image using its path:
    test_label = eigenfaces_model.classify(TEST_IMAGE_PATH)

    # Print classification:
    print("Image classified as: " + test_label)