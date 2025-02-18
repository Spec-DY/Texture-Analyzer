import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from skimage import exposure
import gradio as gr
from tqdm import tqdm


class TextureClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.glcm_classifier = RandomForestClassifier(
            n_estimators=100, random_state=42)
        self.lbp_classifier = RandomForestClassifier(
            n_estimators=100, random_state=42)
        self.classes = ['bricks', 'stone', 'wood']

    def preprocess_image(self, img):
        img = exposure.equalize_hist(img)
        img = (img * 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        return img

    def load_dataset(self, data_path):
        images = []
        labels = []

        # 计算总图片数以设置进度条
        total_images = sum(len(os.listdir(os.path.join(data_path, class_name)))
                           for class_name in self.classes)

        print("Loading and augmenting dataset...")
        pbar = tqdm(total=total_images, desc="Processing images")

        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(data_path, class_name)
            if not os.path.exists(class_path):
                raise ValueError(f"Directory not found: {class_path}")

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (200, 200))
                    img = self.preprocess_image(img)

                    angles = [0, 90, 180, 270]
                    for angle in angles:
                        if angle > 0:
                            matrix = cv2.getRotationMatrix2D(
                                (img.shape[1]/2, img.shape[0]/2), angle, 1)
                            rotated = cv2.warpAffine(
                                img, matrix, (img.shape[1], img.shape[0]))
                            images.append(rotated)
                            labels.append(class_idx)
                        else:
                            images.append(img)
                            labels.append(class_idx)

                    flipped = cv2.flip(img, 1)
                    images.append(flipped)
                    labels.append(class_idx)

                    pbar.update(1)
                else:
                    print(f"Warning: Could not load image {img_path}")

        pbar.close()

        if not images:
            raise ValueError(
                "No images were loaded. Check your data directory structure.")

        return np.array(images), np.array(labels)

    def extract_glcm_features(self, image):
        distances = [1, 2, 3, 4]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        features = []

        for distance in distances:
            glcm = graycomatrix(
                image, [distance], angles, 256, symmetric=True, normed=True)

            features.extend([
                graycoprops(glcm, 'contrast').mean(),
                graycoprops(glcm, 'dissimilarity').mean(),
                graycoprops(glcm, 'homogeneity').mean(),
                graycoprops(glcm, 'energy').mean(),
                graycoprops(glcm, 'correlation').mean(),
                graycoprops(glcm, 'ASM').mean()
            ])

        return np.array(features)

    def extract_lbp_features(self, image):
        features = []
        configs = [
            (1, 8),
            (2, 16),
            (3, 24)
        ]

        for radius, n_points in configs:
            lbp = local_binary_pattern(
                image, n_points, radius, method='uniform')
            hist, _ = np.histogram(
                lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2), density=True)
            features.extend(hist)

        return np.array(features)

    def prepare_features(self, images, feature_type='glcm'):
        features = []
        desc = "Extracting GLCM features" if feature_type == 'glcm' else "Extracting LBP features"

        for img in tqdm(images, desc=desc):
            if feature_type == 'glcm':
                feat = self.extract_glcm_features(img)
            else:  # lbp
                feat = self.extract_lbp_features(img)
            features.append(feat)
        return np.array(features)

    def train(self, data_path):
        images, labels = self.load_dataset(data_path)
        print(f"\nTotal images after augmentation: {len(images)}")

        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.3, random_state=42)

        # Train GLCM classifier
        print("\nProcessing GLCM features...")
        glcm_train_features = self.prepare_features(X_train, 'glcm')
        glcm_test_features = self.prepare_features(X_test, 'glcm')

        print("Scaling GLCM features...")
        glcm_train_features_scaled = self.scaler.fit_transform(
            glcm_train_features)
        glcm_test_features_scaled = self.scaler.transform(glcm_test_features)

        print("\nTraining GLCM classifier with parameter optimization...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10]
        }

        grid_search = GridSearchCV(
            self.glcm_classifier, param_grid, cv=5, n_jobs=-1)
        with tqdm(total=1, desc="GLCM GridSearchCV") as pbar:
            grid_search.fit(glcm_train_features_scaled, y_train)
            pbar.update(1)

        self.glcm_classifier = grid_search.best_estimator_
        glcm_predictions = self.glcm_classifier.predict(
            glcm_test_features_scaled)
        glcm_accuracy = accuracy_score(y_test, glcm_predictions)

        # Train LBP classifier
        print("\nProcessing LBP features...")
        lbp_train_features = self.prepare_features(X_train, 'lbp')
        lbp_test_features = self.prepare_features(X_test, 'lbp')

        print("\nTraining LBP classifier with parameter optimization...")
        grid_search = GridSearchCV(
            self.lbp_classifier, param_grid, cv=5, n_jobs=-1)
        with tqdm(total=1, desc="LBP GridSearchCV") as pbar:
            grid_search.fit(lbp_train_features, y_train)
            pbar.update(1)

        self.lbp_classifier = grid_search.best_estimator_
        lbp_predictions = self.lbp_classifier.predict(lbp_test_features)
        lbp_accuracy = accuracy_score(y_test, lbp_predictions)

        print("\nResults:")
        print(f"GLCM Accuracy: {glcm_accuracy:.3f}")
        print(f"LBP Accuracy: {lbp_accuracy:.3f}")
        print("\nBest parameters for GLCM classifier:", grid_search.best_params_)

        print("\nGLCM Confusion Matrix:")
        print(confusion_matrix(y_test, glcm_predictions))
        print("\nLBP Confusion Matrix:")
        print(confusion_matrix(y_test, lbp_predictions))

    def predict(self, img_path, method='glcm'):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Failed to load image")

        img = cv2.resize(img, (200, 200))
        img = self.preprocess_image(img)

        if method == 'glcm':
            features = self.extract_glcm_features(img)
            features_scaled = self.scaler.transform([features])
            probabilities = self.glcm_classifier.predict_proba(features_scaled)[
                0]
            prediction = self.glcm_classifier.predict(features_scaled)[0]
        else:  # lbp
            features = self.extract_lbp_features(img)
            probabilities = self.lbp_classifier.predict_proba([features])[0]
            prediction = self.lbp_classifier.predict([features])[0]

        result = {
            'prediction': self.classes[prediction],
            'probabilities': {cls: prob for cls, prob in zip(self.classes, probabilities)}
        }
        return result


def create_gradio_interface(classifier):
    def classify_texture(image, method):
        temp_path = "temp_image.jpg"
        cv2.imwrite(temp_path, image)

        try:
            result = classifier.predict(temp_path, method)
            prediction = result['prediction']
            probs = result['probabilities']
            confidence = max(probs.values())

            return (
                prediction,
                confidence,
                probs
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    interface = gr.Interface(
        fn=classify_texture,
        inputs=[
            gr.Image(),
            gr.Radio(["glcm", "lbp"], label="Method", value="glcm")
        ],
        outputs=[
            gr.Label(label="Prediction"),
            gr.Number(label="Confidence"),
            gr.Label(label="Class Probabilities")
        ],
        title="Texture Classifier",
        description="Upload an image to classify its texture as bricks, stone, or wood."
    )

    return interface


if __name__ == "__main__":
    classifier = TextureClassifier()
    classifier.train("data")

    interface = create_gradio_interface(classifier)
    interface.launch(share=True)
