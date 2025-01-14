import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.vgg16 import VGG16, preprocess_input
from PIL import Image
import numpy as np
import pyttsx3
import pickle

# Load your custom-trained VGG16 + LSTM model and tokenizer
@st.cache_resource
def load_custom_captioning_model():
    # Load the trained VGG16 + LSTM model
    model = load_model("my_model.keras")
    
    # Load the tokenizer (assuming it's saved using pickle)
    with open("caption.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    
    # Load VGG16 model for feature extraction
    vgg16_model = VGG16(weights="imagenet", include_top=False, pooling="avg")
    
    return model, tokenizer, vgg16_model

model, tokenizer, vgg16_model = load_custom_captioning_model()

# Function to preprocess image for VGG16
def preprocess_image(image):
    # Resize image to (224, 224) as VGG16 expects this size
    image = image.resize((224, 224))
    
    # Convert the image to an array
    image_array = img_to_array(image)
    
    # Add batch dimension and preprocess
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    
    return image_array

# Function to extract features using VGG16
def extract_features(image):
    image_array = preprocess_image(image)
    features = vgg16_model.predict(image_array)
    return features

# Function to generate captions from an image
def generate_caption(image):
    # Extract features using VGG16
    features = extract_features(image)
    
    # Initialize the caption sequence with the start token
    caption_sequence = [tokenizer.word_index['<start>']]
    
    # Generate caption by iteratively predicting the next word
    for _ in range(220):  # Maximum length of the caption
        # Convert the caption sequence into a padded sequence
        sequence = np.array(caption_sequence).reshape(1, -1)
        
        # Predict the next word using the model
        predicted_word_probs = model.predict([features, sequence])  # Pass both inputs
        
        # Get the index of the word with the highest probability
        predicted_word_index = np.argmax(predicted_word_probs)
        
        # Convert the index to a word
        predicted_word = tokenizer.index_word.get(predicted_word_index, '')
        
        # Stop generating if the end token is predicted
        if predicted_word == '<end>':
            break
        
        # Append the predicted word index to the caption sequence
        caption_sequence.append(predicted_word_index)
    
    # Convert the caption sequence into a string
    caption = ' '.join(
        [tokenizer.index_word[word] for word in caption_sequence if word not in [tokenizer.word_index['<start>'], tokenizer.word_index['<end>']]]
    )
    return caption

# Function to convert text to WAV audio
def text_to_audio_wav(text, output_audio_path):
    engine = pyttsx3.init()
    engine.save_to_file(text, output_audio_path)  # Save as WAV by default
    engine.runAndWait()

# Streamlit App
st.markdown('<h3 style="text-align: center;">Medical Image Caption Generation to Text & Audio </h3>', unsafe_allow_html=True)
# st.title("Medical Image Caption to Text & Audio")

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
