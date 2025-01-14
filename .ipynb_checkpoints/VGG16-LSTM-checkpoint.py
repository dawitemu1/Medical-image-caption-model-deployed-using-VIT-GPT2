import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import pyttsx3
import pickle
import os

# Load your custom-trained VGG16 + LSTM model and tokenizer
@st.cache_resource
def load_custom_captioning_model():
    # Load the trained VGG16 + LSTM model
    model = load_model("my_model.keras.h5")
    
    # Load the tokenizer (assuming it's saved using pickle)
    with open("caption.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    
    return model, tokenizer

model, tokenizer = load_custom_captioning_model()

# Function to preprocess image for VGG16
def preprocess_image(image):
    # Resize image to (224, 224) as VGG16 expects this size
    image = image.resize((224, 224))
    
    # Convert the image to an array
    image_array = img_to_array(image)
    
    # Expand dimensions to match VGG16 input
    image_array = np.expand_dims(image_array, axis=0)
    
    # Normalize the image array
    image_array = image_array / 255.0
    
    return image_array

# Function to generate captions from an image
def generate_caption(image):
    image_array = preprocess_image(image)
    
    # Extract features using the VGG16 model (we assume the model uses this feature extractor)
    features = model.predict(image_array)
    
    # Initialize the caption sequence (start with the token for the start of the caption)
    caption_sequence = [tokenizer.word_index['<start>']]  # Assuming '<start>' is the start token
    
    # Generate caption using the LSTM model (predict the next word iteratively)
    for i in range(30):  # Limit to a maximum of 30 words
        # Convert the caption sequence into a padded sequence
        sequence = np.array(caption_sequence).reshape(1, -1)
        
        # Predict the next word using the model (LSTM)
        predicted_word_probs = model.predict([features, sequence])
        
        # Get the index of the word with the highest probability
        predicted_word_index = np.argmax(predicted_word_probs)
        
        # Convert the index to a word
        predicted_word = tokenizer.index_word.get(predicted_word_index, '')
        
        # Break if the end token is predicted
        if predicted_word == '<end>':
            break
        
        # Append the predicted word to the caption sequence
        caption_sequence.append(predicted_word_index)
    
    # Decode the caption sequence back to words
    caption = ' '.join([word for word in caption_sequence if word not in [tokenizer.word_index['<start>'], tokenizer.word_index['<end>']]])
    return caption

# Function to convert text to WAV audio
def text_to_audio_wav(text, output_audio_path):
    engine = pyttsx3.init()
    engine.save_to_file(text, output_audio_path)  # Save as WAV by default
    engine.runAndWait()

# Streamlit App
st.title("Image Caption to WAV Audio")

# Image Upload
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Variables to store the caption and audio path
caption = None
audio_path = "generated_audio.wav"

if uploaded_image:
    # Display uploaded image
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Button to generate caption
    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            try:
                caption = generate_caption(image)
                st.success("Caption generated successfully!")
                st.write(f"**Generated Caption:** {caption}")
            except Exception as e:
                st.error(f"Error generating caption: {e}")

    # Button to generate audio (only enabled if a caption exists)
    if caption:
        if st.button("Generate Audio"):
            with st.spinner("Converting caption to audio..."):
                try:
                    text_to_audio_wav(caption, audio_path)
                    st.success("WAV audio generated successfully!")

                    # Play the audio
                    with open(audio_path, "rb") as audio_file:
                        st.audio(audio_file.read(), format="audio/wav")

                    # Option to download the audio
                    st.download_button(
                        label="Download WAV Audio",
                        data=open(audio_path, "rb").read(),
                        file_name="caption_audio.wav",
                        mime="audio/wav",
                    )
                except Exception as e:
                    st.error(f"Error converting text to audio: {e}")
