import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay
import cv2
import matplotlib.pyplot as plt


class ActiveAppearanceModel:
    def __init__(self, shapes, textures, texture_size):
        """
        Initialize the Active Appearance Model.
        :param shapes: List of aligned shapes (n_samples x n_points x 2).
        :param textures: List of associated textures, normalized to the mean shape.
        :param texture_size: Size of texture images (width, height).
        """
        self.shapes = shapes
        self.textures = textures
        self.texture_size = texture_size

    def build_shape_model(self):
        """
        Build a statistical model for shapes.
        """
        print("Building shape model...")
        shape_matrix = np.array([shape.flatten() for shape in self.shapes])
        
        # PCA on shapes
        self.shape_pca = PCA()
        self.shape_pca.fit(shape_matrix)
        
        self.shape_modes = self.shape_pca.components_
        self.shape_mean_vector = self.shape_pca.mean_
        self.mean_shape = self.shape_mean_vector.reshape(-1, 2)
        
        print(f"Shape model built with {len(self.shape_modes)} principal modes.")

        # Plot mean shape landmarks
        plt.figure()
        plt.scatter(self.mean_shape[:, 0], self.mean_shape[:, 1], color='red')
        plt.title("Mean Shape Landmarks")
        plt.axis('equal')
        plt.show()

    def build_texture_model(self):
        """
        Build a statistical model for textures at multiple resolutions using a Gaussian pyramid.
        """
        print("Building texture model...")
        self.texture_pyramids = []
        self.texture_pca_models = []
        
        # Create Gaussian pyramids for each texture
        pyramid_levels = 3
        for level in range(pyramid_levels):
            pyramid_textures = [self._get_pyramid_level(texture, level) for texture in self.textures]
            pyramid_matrix = np.array([texture.flatten() for texture in pyramid_textures])
            
            # Normalize textures
            normalized_textures = (pyramid_matrix - pyramid_matrix.mean(axis=1, keepdims=True)) / \
                                   (pyramid_matrix.std(axis=1, keepdims=True) + 1e-6)
            
            # PCA for each pyramid level
            texture_pca = PCA()
            texture_pca.fit(normalized_textures)
            
            self.texture_pyramids.append(pyramid_textures)
            self.texture_pca_models.append(texture_pca)
            print(f"Texture model for pyramid level {level} built with {len(texture_pca.components_)} modes.")

    def warp_texture(self, mean_shape, target_shape, texture):
        """
        Warp the texture from the mean shape to the target shape using triangulation-based interpolation.
        """
        texture_image = texture.reshape(self.texture_size)
        
        # Perform Delaunay triangulation on the mean shape
        triangulation = Delaunay(mean_shape)
        
        warped_texture = np.zeros_like(texture_image)
        
        for simplex in triangulation.simplices:
            src_points = mean_shape[simplex].astype(np.float32)
            dst_points = target_shape[simplex].astype(np.float32)
            
            # Compute affine transform
            affine_transform = cv2.getAffineTransform(src_points, dst_points)
            
            # Mask for the current triangle
            mask = np.zeros(texture_image.shape, dtype=np.uint8)
            cv2.fillConvexPoly(mask, dst_points.astype(np.int32), 1)
            
            # Warp the triangle region
            warped_region = cv2.warpAffine(texture_image, affine_transform, texture_image.shape[::-1])
            
            # Blend the warped region into the result
            warped_texture[mask == 1] = warped_region[mask == 1]
        
        # Plot warped texture
        plt.figure()
        plt.imshow(warped_texture, cmap='gray')
        plt.title("Warped Texture Example")
        plt.show()

        return warped_texture

    def _get_pyramid_level(self, texture, level):
        """
        Get a texture at a given pyramid level using Gaussian downsampling.
        """
        image = texture.reshape(self.texture_size)
        for _ in range(level):
            image = cv2.pyrDown(image)
        return image

    def build_combined_model(self):
        """
        Build a combined model for shape and texture.
        """
        print("Building combined model...")
        # Shape parameters from PCA
        shape_params = self.shape_pca.transform(
            np.array([shape.flatten() for shape in self.shapes])
        )
        
        # Texture parameters from the top pyramid level
        texture_params = self.texture_pca_models[-1].transform(
            [texture.flatten() for texture in self.texture_pyramids[-1]]
        )
        
        # Combine shape and texture parameters
        combined_data = np.hstack((shape_params, texture_params))
        
        # PCA on combined data
        self.combined_pca = PCA()
        self.combined_pca.fit(combined_data)
        
        self.combined_modes = self.combined_pca.components_
        self.combined_mean_vector = self.combined_pca.mean_
        print(f"Combined model built with {len(self.combined_modes)} principal modes.")

    def generate_shape(self, shape_params):
        """
        Generate a shape from given shape parameters.
        """
        shape_vector = self.shape_mean_vector + np.dot(self.shape_modes.T, shape_params)
        return shape_vector.reshape(-1, 2)

    def generate_texture(self, texture_params, level=0):
        """
        Generate a texture from given texture parameters at a specific pyramid level.
        """
        texture_pca = self.texture_pca_models[level]
        texture_mean = texture_pca.mean_
        return texture_mean + np.dot(texture_pca.components_.T, texture_params)

    def generate_combined(self, subject_idx):
        """
        Generate a shape and texture for a specific subject in the dataset.
        """
        shape_params = self.shape_pca.transform([self.shapes[subject_idx].flatten()])[0]
        texture_params = self.texture_pca_models[-1].transform([self.texture_pyramids[-1][subject_idx].flatten()])[0]
        
        shape = self.generate_shape(shape_params)
        texture = self.generate_texture(texture_params, level=-1)
        return shape, texture


