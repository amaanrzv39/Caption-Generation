import streamlit as st
from prediction import predict
from PIL import Image


st.title("Image Captioning")

# any type jpg, png, jpeg
img = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# if image is uploaded
if st.button("Generate Caption") and img is not None:
    # display the image
    img = Image.open(img).convert("RGB")
    col1, col2 = st.columns(2)
    
    with col1:
        # Resize imgage on display
        st.image(img, caption="Uploaded Image")
    
    # predict the caption
    caption = predict(img)
    
    with col2:
        st.write('<h3 style="color:#B34088;">Generated Caption</h3>', unsafe_allow_html=True) 
        # Write Caption in yellow color
        st.write(f'<p style="color:#f9ca24";>{caption}</p>', unsafe_allow_html=True)

