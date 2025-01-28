import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import procrustes

class ActiveAppearanceModel:
    def __init__(self, shapes, textures):
        """
        Initialise le modèle d'apparence active.
        :param shapes: Liste des formes alignées (n_samples x n_points x 2).
        :param textures: Liste des textures associées, normalisées à une forme moyenne.
        """
        self.shapes = shapes  # Données de forme
        self.textures = textures  # Données de texture

    def build_shape_model(self):
        """
        Construit un modèle statistique pour les formes.
        """
        print("Construction du modèle de forme...")
        # Étape 1 : Alignement des formes avec l'analyse Procruste
        aligned_shapes = []
        reference_shape = self.shapes[0]
        for shape in self.shapes:
            _, aligned, _ = procrustes(reference_shape, shape)
            aligned_shapes.append(aligned)
        self.mean_shape = np.mean(aligned_shapes, axis=0)
        
        # Étape 2 : Conversion des formes en vecteurs
        shape_matrix = np.array([shape.flatten() for shape in aligned_shapes])
        
        # Étape 3 : Analyse en composantes principales (ACP)
        self.shape_pca = PCA()
        self.shape_pca.fit(shape_matrix)
        
        self.shape_modes = self.shape_pca.components_
        self.shape_mean_vector = self.shape_pca.mean_
        print(f"Modèle de forme construit avec {len(self.shape_modes)} modes principaux.")

    def build_texture_model(self):
        """
        Construit un modèle statistique pour les textures.
        """
        print("Construction du modèle de texture...")
        # Étape 1 : Normalisation des textures
        normalized_textures = []
        for texture in self.textures:
            norm_texture = (texture - np.mean(texture)) / np.std(texture)
            normalized_textures.append(norm_texture)
        self.mean_texture = np.mean(normalized_textures, axis=0)
        
        # Étape 2 : Conversion des textures en vecteurs
        texture_matrix = np.array([texture.flatten() for texture in normalized_textures])
        
        # Étape 3 : Analyse en composantes principales (ACP)
        self.texture_pca = PCA()
        self.texture_pca.fit(texture_matrix)
        
        self.texture_modes = self.texture_pca.components_
        self.texture_mean_vector = self.texture_pca.mean_
        print(f"Modèle de texture construit avec {len(self.texture_modes)} modes principaux.")

    def build_combined_model(self):
        """
        Construit un modèle combiné forme + texture.
        """
        print("Construction du modèle combiné...")
        # Étape 1 : Obtenir les paramètres des formes et des textures
        shape_params = self.shape_pca.transform(
            np.array([shape.flatten() for shape in self.shapes])
        )
        texture_params = self.texture_pca.transform(
            np.array([texture.flatten() for texture in self.textures])
        )
        
        # Étape 2 : Combinaison des paramètres
        combined_data = np.hstack((shape_params, texture_params))
        
        # Étape 3 : ACP sur les données combinées
        self.combined_pca = PCA()
        self.combined_pca.fit(combined_data)
        
        self.combined_modes = self.combined_pca.components_
        self.combined_mean_vector = self.combined_pca.mean_
        print(f"Modèle combiné construit avec {len(self.combined_modes)} modes principaux.")

    def generate_shape(self, b_s):
        """
        Génère une forme à partir des paramètres b_s.
        :param b_s: Paramètres des modes de forme.
        :return: Coordonnées des points de forme (n_points x 2).
        """
        shape_vector = self.shape_mean_vector + np.dot(self.shape_modes.T, b_s)
        return shape_vector.reshape(-1, 2)

    def generate_texture(self, b_g):
        """
        Génère une texture à partir des paramètres b_g.
        :param b_g: Paramètres des modes de texture.
        :return: Texture générée sous forme de vecteur.
        """
        return self.texture_mean_vector + np.dot(self.texture_modes.T, b_g)

    def generate_combined(self, c):
        """
        Génère une forme et une texture combinées à partir des paramètres c.
        :param c: Paramètres combinés (forme + texture).
        :return: (forme, texture)
        """
        n_shape_modes = len(self.shape_modes)
        b_s = c[:n_shape_modes]
        b_g = c[n_shape_modes:]
        
        shape = self.generate_shape(b_s)
        texture = self.generate_texture(b_g)
        return shape, texture


# Exemple d'utilisation
if __name__ == "__main__":
    # Simuler des données de formes (100 échantillons, 68 points, 2 coordonnées)
    n_samples = 100
    n_points = 68
    shapes = [np.random.rand(n_points, 2) for _ in range(n_samples)]
    
    # Simuler des textures associées (par exemple, 256 x 256 pixels par échantillon)
    texture_size = 256 * 256
    textures = [np.random.rand(texture_size) for _ in range(n_samples)]
    
    # Créer et construire le modèle
    aam = ActiveAppearanceModel(shapes, textures)
    aam.build_shape_model()
    aam.build_texture_model()
    aam.build_combined_model()
    
    # Générer une forme et une texture
    example_params = np.random.rand(len(aam.combined_modes))  # Paramètres combinés aléatoires
    generated_shape, generated_texture = aam.generate_combined(example_params)
    
    print("Forme générée :", generated_shape)
    print("Texture générée :", generated_texture[:10])  # Afficher les 10 premiers pixels
