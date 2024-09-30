import streamlit as st
from transformers import AutoModel, AutoTokenizer, MarianMTModel, MarianTokenizer
from PIL import Image
import tempfile
import os
import easyocr
import re
import torch

# Define a function to load models
@st.cache_resource
def load_model():
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('stepfun-ai/GOT-OCR2_0', trust_remote_code=True)
    model = AutoModel.from_pretrained(
        'stepfun-ai/GOT-OCR2_0', 
        trust_remote_code=True, 
        low_cpu_mem_usage=True, 
        use_safetensors=True, 
        pad_token_id=tokenizer.eos_token_id
    )
    model.eval()  # Set model to evaluation mode
    return tokenizer, model

# Load the models once at the start
tokenizer, model = load_model()

# Check if GPU is available, else default to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)  # Move model to appropriate device
st.write(f"Using device: {device}")

# Load EasyOCR reader with English and Hindi language support
reader = easyocr.Reader(['en', 'hi'])  # 'en' for English, 'hi' for Hindi

# Load MarianMT translation model for Hindi to English translation
translation_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-hi-en')
translation_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-hi-en')

# Define a function for keyword highlighting
def highlight_keywords(text, keyword):
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    highlighted_text = pattern.sub(lambda match: f"**{match.group(0)}**", text)
    return highlighted_text

# Streamlit App Title
st.title("OCR with GOT-OCR2 (English & Hindi Translation) and Keyword Search")

# File uploader for image input
image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if image_file is not None:
    # Display the uploaded image
    image = Image.open(image_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file.write(image_file.getvalue())
        temp_file_path = temp_file.name

    # Button to run OCR
    if st.button("Run OCR"):
        # Use GOT-OCR2 model for plain text OCR (structured documents)
        with torch.no_grad():
            extracted_text = "some OCR processed text"  # Placeholder, replace with actual OCR result
            # Perform OCR and other processing
            # Tokenize the extracted text
            inputs = tokenizer(extracted_text, return_tensors="pt", truncation=True, padding=True)
            input_ids = inputs['input_ids'].to(device)  # Use .to(device)

            # Example model processing
            # Replace with actual model inference code
            # res_plain = model(input_ids)

            # Use EasyOCR for both English and Hindi text recognition
            result_easyocr = reader.readtext(temp_file_path, detail=0)

            # Display the results
            st.subheader("Detected Text using EasyOCR (English and Hindi):")
            extracted_text = " ".join(result_easyocr)  # Combine the list of text results
            st.write(extracted_text)

            # Translate Hindi text to English using MarianMT (optional step)
            st.subheader("Translated Hindi Text to English:")
            translated_text = []
            for sentence in result_easyocr:
                if sentence:  # Assuming non-empty text is translated
                    tokenized_text = translation_tokenizer([sentence], return_tensors="pt", truncation=True)
                    tokenized_text = {key: val.to(device) for key, val in tokenized_text.items()}  # Move tensors to device
                    translation = translation_model.generate(**tokenized_text)
                    translated_sentence = translation_tokenizer.decode(translation[0], skip_special_tokens=True)
                    translated_text.append(translated_sentence)

            st.write(" ".join(translated_text))

            # Search functionality
            keyword = st.text_input("Enter keyword to search in extracted text:")

            if keyword:
                st.subheader("Search Results:")
                highlighted_text = highlight_keywords(extracted_text, keyword)
                st.markdown(highlighted_text)

        # Clean up the temporary file after use
        os.remove(temp_file_path)
