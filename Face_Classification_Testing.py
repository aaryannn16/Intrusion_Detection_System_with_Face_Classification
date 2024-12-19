import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

# Load the pre-trained model
model = load_model('New_face_classification_model.h5')

# Load and preprocess an image for prediction
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224,3))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Function to predict the person in the image
def classify_face(img_path):
    img_array = load_and_preprocess_image(img_path)
    
    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])  # Get the index of the highest probability
    
    # Assuming you have a mapping of class indices to names of people
    class_labels = ['Aryan','Garvit','Tanisha']  # Replace with actual labels
    
    print(f"Predicted Person: {class_labels[predicted_class]} with confidence {np.max(predictions[0]) * 100:.2f}%")

if __name__ == "__main__":
    # Get the image path from the user
    img_path = input("Enter the path of the image to classify: ")
    
    # Classify the face in the input image
    classify_face(img_path)
