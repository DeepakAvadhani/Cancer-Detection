import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.preprocessing import image
import cv2

url = "https://github.com/DeepakAvadhani/Cancer-Detection/blob/main/skin_cancer_detection_model.h5"
filename = "skin_cancer_detection_model.h5"

# Download if not exists
if not os.path.exists(filename):
    r = requests.get(url, allow_redirects=True)
    with open(filename, 'wb') as f:
        f.write(r.content)


# Load the trained model
def load_model():
    def top_2_accuracy(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=2)

    def top_3_accuracy(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=3)

    model = tf.keras.models.load_model(filename, custom_objects={'top_2_accuracy': top_2_accuracy, 'top_3_accuracy': top_3_accuracy})
    return model

# Define the labels
labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Function to preprocess the input image
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)
    return img_array

# Function to make prediction and return confidence
def predict_skin_cancer(img):
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    confidence = np.max(prediction)  # Maximum probability value
    predicted_class_index = np.argmax(prediction)
    predicted_class = labels[predicted_class_index]
    return predicted_class, confidence

# Function to predict whether the person has skin cancer or not
def predict_skin_condition(img):
    predicted_class, confidence = predict_skin_cancer(img)
    if predicted_class == 'nv':
        return "Person doesn't have skin cancer", confidence
    else:
        cancer_info = get_cancer_info(predicted_class)
        return f"Person has skin cancer ({predicted_class})", confidence, cancer_info

def get_cancer_info(predicted_class):
    cancer_info = {
        'akiec': "Actinic Keratoses (akiec) is a type of precancerous skin growth that can develop into squamous cell carcinoma.\nIt typically appears as rough, scaly patches on the skin, often on areas frequently exposed to the sun.",
        'bcc': "Basal Cell Carcinoma (bcc) is a common type of skin cancer that arises from the basal cells.\nIt usually appears as a flesh-colored or pinkish bump on the skin, often with visible blood vessels or a central depression.",
        'bkl': "Benign Keratosis-Like Lesions (bkl) are non-cancerous growths that resemble a form of skin cancer.\nThey can include seborrheic keratoses, solar lentigines, and other benign growths that may mimic melanoma or other skin cancers.",
        'df': "Dermatofibroma (df) is a benign skin lesion that can resemble a malignant tumor.\nIt usually appears as a firm, raised bump on the skin, often brownish or reddish in color, and may have a dimple in the center when pinched.",
        'mel': "Melanoma (mel) is a type of skin cancer that develops from melanocytes, the pigment-producing cells of the skin.\nIt often appears as a dark, irregularly shaped mole or lesion on the skin, and it can spread rapidly if not treated early.",
        'vasc': "Vascular Lesions (vasc) are abnormalities of the blood vessels in the skin.\nThey can include various conditions such as hemangiomas, port-wine stains, and spider veins, which may or may not be associated with cancer."
    }
    return cancer_info.get(predicted_class, "No information available for this type of cancer.")

# Load the model
model = load_model()

# Streamlit app
st.title("Skin Cancer Detection")

# Option to choose between uploading an image or capturing from camera
option = st.radio("Choose an option:", ("Upload an image", "Capture from camera"))

if option == "Upload an image":
    # Option to upload an image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        # Display the uploaded image
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        # Make prediction
        result, confidence, cancer_info = predict_skin_condition(img)
        st.write("Prediction Result:", result)
        st.write("Confidence:", confidence)
        if cancer_info:
            st.write("Additional Information about the Cancer Type:")
            st.write(cancer_info)
else:
    # Capture from camera
    st.write("Preparing to capture image from camera...")
    
    # Function to capture image from camera
    def capture_image():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Unable to open camera.")
            return None
        ret, frame = cap.read()
        if ret:
            return frame
        else:
            st.error("Error: Unable to capture image from camera.")
            return None

    # Capture image from camera
    captured_frame = capture_image()
    if captured_frame is not None:
        # Convert OpenCV frame to PIL image
        pil_image = Image.fromarray(cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB))
        
        # Display the captured image
        st.image(pil_image, caption="Captured Image", use_column_width=True)
        
        # Make prediction
        result, confidence, cancer_info = predict_skin_condition(pil_image)
        st.write("Prediction Result:", result)
        st.write("Confidence:", confidence)
        if cancer_info:
            st.write("Additional Information about the Cancer Type:")
            st.write(cancer_info)

