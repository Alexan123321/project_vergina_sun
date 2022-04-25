from dataclasses import dataclass
import numpy as np
from PIL import Image #Used in preprocesser

@dataclass
class Eigenfaces:
    '''
    This class represents the Eigenfaces module.
    '''
    is_trained: bool
    eigenfaces: np.array([])
    mean_face: np.array([])

    def __init__(self):
        self.is_trained = False
    
    def train(self, training_images: np.array([])) -> None:
        '''
        SUMMARY: This method follows the Eigenfaces algorithm, thus it 1) normalizes the
        training images, 2) calculates the mean face and 3) calculates the eigenfaces.
        PARAMETERS: A numpy list of training images and a numpy list of labels.
        RETURNS: None.
        '''
        # Eigenfaces procedure initialization:
        # Step 1: Pre-process training images:
        processed_training_images = self._image_preprocesser(training_images)

        # Step 2: Calculate the mean face, and store it in the model:
        self.mean_face = self._mean(processed_training_images)

        # Step 3: Calculate the Eigenfaces using PCA, and store it in the model:
        self.eigenfaces = self._pca(processed_training_images)

        # Update the training attribute:
        self.is_trained = True

    #TODO: implement
    def classify(self, image: np.array([])) -> np.array([]):
        labels = np.array([])
        return labels

    def _image_preprocesser(self, images: np.array([])) -> np.array([]):
        '''
        SUMMARY: This method follows the Eigenfaces algorithm, thus it 1) normalizes the
        training images, 2) calculates the mean face and 3) calculates the eigenfaces.
        PARAMETERS: A numpy list of training images and labels.
        RETURNS: None.
        '''
        #Temporary container to store processed images:
        processed_images = np.array([])
        # Default image size declaration:
        image_default_size = [64, 64] 
        # Processing loop:
        for image in images: 
            #Convert current image to greyscale:
            image = image.convert("L")
            #Resize to default size using antialiasing:
            image = image.resize(image_default_size, Image.ANTIALIAS)
            #Convert image to numpy array with data type uint8
            image = np.asarray(image, dtype=np.uint8) 
            #Append the processed image to temporary container:
            np.append(processed_images, image)
        #Return processed images:
        return processed_images

    def _mean(self, x: np.array([])) -> np.array([]):
        '''
        SUMMARY: This method calculates the mean of a list using Goldschmidt's algorithm for division.
        PARAMETERS: A numpy list.
        RETURNS: A numpy list.
        '''
        # Calculate the sum of x:
        __sum = self._sum(x)
        # Determine the number of elements in x:
        n = len(x)
        # Calculate the mean of x:
        mean = self._goldschmidt_division(__sum, n)
        # Return the mean:
        return mean
    
    def _sum(self, x: np.array([])) -> np.array([]):
        '''
        SUMMARY: This method calculates the sum of a list. 
        PARAMETERS: A numpy list.
        RETURNS: A numpy list.
        '''
        # Initialize a temporary sum and set it equal to 0:
        __sum = 0
        # Iterate over all elements in the parameter list:
        for i in x:
            __sum = __sum + x[i]
        # Return temporary sum:
        return __sum

    def _goldschmidt_division(self, a, b) -> float:
        '''
        SUMMARY: This method calculates the fraction of a divided by b. 
        PARAMETERS: A nominator, a, and a denominator, b.
        RETURNS: A float, a.
        '''
        no_iterations = 60
        #r = 1/b
        r = self._newton_inverse(b)
        for _ in range(no_iterations):
            a = r*a
            b = r*b
            r = 2 + -1 * b
        return a

    def _newton_inverse(self, a) -> float:
        '''
        SUMMARY: This method calculates the inverse of a number, a. 
        PARAMETERS: A number, a.
        RETURNS: A float, inv_a.
        '''
        no_iterations = 4
        inv_a = a
        for _ in range(no_iterations):
            inv_a = 0.5 * (inv_a + a / inv_a) 
        return inv_a

    #TODO: implement
    def _pca(self, images) -> np.array([]):
        num_components = len(X)
        mu = X.mean(axis = 0)
        X = X - mu
        C = np.dot (X,X.T) # Covariance Matrix
        [ eigenvalues , eigenvectors ] = pow_eig_comb(C)
        eigenvectors = np.dot(X.T, eigenvectors )
        for i in range(0,len(X)):
            eigenvectors [:,i] = eigenvectors [:,i]/ np.linalg.norm( eigenvectors [:,i])
        # sort eigenvectors descending by their eigenvalue
        idx = np.argsort (- eigenvalues )
        eigenvalues = eigenvalues [idx ]
        eigenvectors = eigenvectors [:, idx ]
        num_components = get_number_of_components_to_preserve_variance(eigenvalues)
        # select only num_components
        eigenvalues = eigenvalues [0: num_components ].copy ()
        eigenvectors = eigenvectors [: ,0: num_components ].copy ()
        return [ eigenvalues , eigenvectors , mu]  