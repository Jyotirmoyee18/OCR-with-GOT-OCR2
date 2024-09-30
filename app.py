import streamlit as st
from transformers import AutoModel, AutoTokenizer, MarianMTModel, MarianTokenizer
from PIL import Image
import tempfile
import os
import easyocr
import re

# Load EasyOCR reader with English and Hindi language support
reader = easyocr.Reader(['en', 'hi'])  # 'en' for English, 'hi' for Hindi

# Load the GOT-OCR2 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
model = model.eval().cuda()

# Load MarianMT translation model for Hindi to English translation
translation_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-hi-en')
translation_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-hi-en')

# Define a function for keyword highlighting
def highlight_keywords(text, keyword):
    # Escape keyword for regex to avoid issues with special characters
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
        res_plain = model.chat(tokenizer, temp_file_path, ocr_type='ocr')

        # Perform formatted text OCR
        res_format = model.chat(tokenizer, temp_file_path, ocr_type='format')

        # Use EasyOCR for both English and Hindi text recognition
        result_easyocr = reader.readtext(temp_file_path, detail=0)

        # Display the results
        st.subheader("Plain Text OCR Results (English):")
        st.write(res_plain)

        st.subheader("Formatted Text OCR Results:")
        st.write(res_format)

        st.subheader("Detected Text using EasyOCR (English and Hindi):")
        extracted_text = " ".join(result_easyocr)  # Combine the list of text results
        st.write(extracted_text)

        # Translate Hindi text to English using MarianMT (optional step)
        st.subheader("Translated Hindi Text to English:")
        translated_text = []
        for sentence in result_easyocr:
            # Detect if the text is in Hindi (you can customize this based on text properties)
            if sentence:  # Assuming non-empty text is translated
                tokenized_text = translation_tokenizer([sentence], return_tensors="pt", truncation=True)
                translation = translation_model.generate(**tokenized_text)
                translated_sentence = translation_tokenizer.decode(translation[0], skip_special_tokens=True)
                translated_text.append(translated_sentence)
        
        st.write(" ".join(translated_text))

        # Additional OCR types using GOT-OCR2
        res_fine_grained = model.chat(tokenizer, temp_file_path, ocr_type='ocr', ocr_box='')
        st.subheader("Fine-Grained OCR Results:")
        st.write(res_fine_grained)

        # Render formatted OCR to HTML
        res_render = model.chat(tokenizer, temp_file_path, ocr_type='format', render=True, save_render_file='./demo.html')
        st.subheader("Rendered OCR Results (HTML):")
        st.write(res_render)

        # Search functionality
        keyword = st.text_input("Enter keyword to search in extracted text:")

        if keyword:
            st.subheader("Search Results:")
            # Highlight the matching sections in the extracted text
            highlighted_text = highlight_keywords(extracted_text, keyword)
            st.markdown(highlighted_text)

        # Clean up the temporary file after use
        os.remove(temp_file_path)

# Note: No need for if __name__ == "__main__": st.run()
