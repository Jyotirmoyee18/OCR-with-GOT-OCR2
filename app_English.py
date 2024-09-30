import streamlit as st
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import tempfile
import os



# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
model = model.eval().cuda()

# Streamlit App Title
st.title("OCR with GOT-OCR2")

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
        # Perform plain text OCR
        res_plain = model.chat(tokenizer, temp_file_path, ocr_type='ocr')

        # Perform formatted text OCR
        res_format = model.chat(tokenizer, temp_file_path, ocr_type='format')

        # Display the results
        st.subheader("Plain Text OCR Results:")
        st.write(res_plain)
        

        
        st.subheader("Formatted Text OCR Results:")
        st.write(res_format)

        # You can add more OCR types as needed
        # e.g., fine-grained OCR
        res_fine_grained = model.chat(tokenizer, temp_file_path, ocr_type='ocr', ocr_box='')
        st.subheader("Fine-Grained OCR Results:")
        st.write(res_fine_grained)

        # Render formatted OCR to HTML
        res_render = model.chat(tokenizer, temp_file_path, ocr_type='format', render=True, save_render_file='./demo.html')
        st.subheader("Rendered OCR Results (HTML):")
        st.write(res_render)

    # Clean up the temporary file after use
    os.remove(temp_file_path)

# Note: No need for if __name__ == "__main__": st.run()