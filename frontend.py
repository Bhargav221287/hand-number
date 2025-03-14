import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import io
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

# Assume these are imported from your existing code
# from your_model_module import model_with_regularization_22, predict_label

# Define the predict_label function from your backend
def predict_label(image_np):
    """
    Takes a single-row NumPy array (shape: (1, 784)), 
    predicts the label, and returns it.
    Args:
    image_np (numpy array): A (1, 784) shaped input image.
    Returns:
    int: Predicted label
    """
    # Ensure input shape is (1, 784)
    assert image_np.shape == (1, 784), "Input should be a (1, 784) numpy array."
    # Predict
    y_pred_logits = model_with_regularization_22.predict(image_np, verbose=0)  
    y_pred = np.argmax(y_pred_logits, axis=1)[0]  # Get the predicted class
    return y_pred

# Function to preprocess the drawn image
def preprocess_image(image):
    # Convert to grayscale
    image = image.convert('L')
    # Resize to 28x28
    image = image.resize((28, 28))
    # Invert colors if needed (assuming dark digit on light background)
    image = ImageOps.invert(image)
    # Convert to numpy array
    img_array = np.array(image)
    # Normalize pixel values
    img_array = img_array / 255.0
    # Flatten to 1x784
    img_array = img_array.reshape(1, 784)
    return img_array

def main():
    st.title("Handwritten Digit Recognition")
    st.write("Draw a digit (0-9) below and the model will predict what it is.")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Draw Digit", "Upload Image"])
    
    with tab1:
        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        # Add buttons for prediction and clearing
        col1, col2 = st.columns(2)
        predict_button = col1.button("Predict Digit")
        clear_button = col2.button("Clear Canvas")
        
        if clear_button:
            # This will trigger a rerun with a fresh canvas
            st.experimental_rerun()
            
        if predict_button:
            if canvas_result.image_data is not None:
                # Convert the drawn image to PIL format
                image = Image.fromarray(canvas_result.image_data.astype('uint8'))
                
                # Display the preprocessed image
                st.write("Preprocessed 28x28 Image:")
                preprocessed = Image.fromarray(
                    (preprocess_image(image).reshape(28, 28) * 255).astype(np.uint8)
                )
                st.image(preprocessed, width=140)
                
                # Get the preprocessed image as a numpy array
                img_array = preprocess_image(image)
                
                try:
                    # Comment out actual prediction for now, as we don't have the model
                    # prediction = predict_label(img_array)
                    # st.success(f"Prediction: {prediction}")
                    
                    # Placeholder for actual prediction
                    st.success("In a real app, this would call your predict_label function")
                    st.info("The model would receive a numpy array with shape (1, 784)")
                    
                    # Show the first few values of the array
                    st.write("Preview of the input to predict_label function:")
                    st.write(img_array[0][:20])  # Show first 20 values
                except Exception as e:
                    st.error(f"Prediction error: {e}")
            else:
                st.warning("Please draw a digit first!")
                
    with tab2:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=280)
            
            # Convert the image to the required format
            img_array = preprocess_image(image)
            
            # Display the preprocessed image
            st.write("Preprocessed 28x28 Image:")
            preprocessed = Image.fromarray(
                (img_array.reshape(28, 28) * 255).astype(np.uint8)
            )
            st.image(preprocessed, width=140)
            
            if st.button("Predict Uploaded Image"):
                try:
                    # Comment out actual prediction for now
                    # prediction = predict_label(img_array)
                    # st.success(f"Prediction: {prediction}")
                    
                    # Placeholder for actual prediction
                    st.success("In a real app, this would call your predict_label function")
                    st.info("The model would receive a numpy array with shape (1, 784)")
                    
                    # Show the first few values of the array
                    st.write("Preview of the input to predict_label function:")
                    st.write(img_array[0][:20])  # Show first 20 values
                except Exception as e:
                    st.error(f"Prediction error: {e}")

if __name__ == "__main__":
    main()
