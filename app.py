import numpy as np
import cv2
import streamlit as st
from PIL import Image

def colorize_image(img):
    # Convert to grayscale and then to RGB
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    
    # Load the neural network model and cluster centers
    prototxt_path = "models/models_colorization_deploy_v2.prototxt"
    model_path = "models/colorization_release_v2.caffemodel"
    pts_path = "models/pts_in_hull.npy"
    
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    pts = np.load(pts_path)
    pts = pts.transpose().reshape(2, 313, 1, 1)
    
    # Set the network layers
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    
    # Prepare the image
    scaled_img = gray_rgb_img.astype("float32") / 255.0
    lab_img = cv2.cvtColor(scaled_img, cv2.COLOR_RGB2LAB)
    resized_img = cv2.resize(lab_img, (224, 224))
    L_channel = cv2.split(resized_img)[0]
    L_channel -= 50
    
    # Perform colorization
    net.setInput(cv2.dnn.blobFromImage(L_channel))
    ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))
    
    L_channel = cv2.split(lab_img)[0]
    colorized_img = np.concatenate((L_channel[:, :, np.newaxis], ab_channel), axis=2)
    colorized_img = cv2.cvtColor(colorized_img, cv2.COLOR_LAB2RGB)
    colorized_img = np.clip(colorized_img, 0, 1)
    colorized_img = (255 * colorized_img).astype("uint8")
    
    return colorized_img

# Streamlit UI
st.set_page_config(
    page_title="Image Colorizer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .main {
        padding: 20px;
    }
    .title h1 {
        color: #333;
        text-align: center;
        margin-bottom: 20px;
    }
    .description {
        font-size: 18px;
        color: #555;
        text-align: center;
        margin-bottom: 20px;
    }
    .uploader {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    .uploader .file-upload-wrapper {
        width: 50%;
    }
    .output-section {
        display: flex;
        justify-content: center;
        gap: 50px;
        margin-top: 30px;
    }
    .output-section .output {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='title'><h1>Colorize Your Black & White Images</h1></div>", unsafe_allow_html=True)
st.markdown("<div class='description'>This application uses deep learning to bring your black and white images to life. Simply upload an image to get started.</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "png"], key="uploader")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image)
    
    colorized_image = colorize_image(img)
    
    st.markdown("<div class='output-section'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='output'><h3>Original Image</h3></div>", unsafe_allow_html=True)
        st.image(image, use_column_width=True)
    
    with col2:
        st.markdown("<div class='output'><h3>Colorized Image</h3></div>", unsafe_allow_html=True)
        st.image(colorized_image, use_column_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.success("Image colorized successfully!")
else:
    st.markdown("<div class='description'>Please upload a black and white image to see the magic of colorization.</div>", unsafe_allow_html=True)