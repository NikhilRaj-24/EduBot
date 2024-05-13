import streamlit as st
import google.generativeai as genai
import google.ai.generativelanguage as glm
from PIL import Image

API_KEY = "AIzaSyBoxbF4cfK3qhi2kcobMHGYgn7TE-l1YG0"

genai.configure(api_key=API_KEY)
st.set_page_config(page_title="Image Analysis" ,page_icon='📸', layout='centered', initial_sidebar_state='collapsed')
st.header('Image Analyzer')
uploaded_file = st.file_uploader('Choose an Image file', accept_multiple_files=False, type=['jpg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    bytes_data = uploaded_file.getvalue()

generate = st.button('Generate!')

if generate:
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content(glm.Content(parts = [glm.Part(text="Write a short, you know this? what is this?"),glm.Part(inline_data=glm.Blob(mime_type='image/jpeg', data=bytes_data )),], ),stream=True)
    response.resolve()
    st.write(response.text)