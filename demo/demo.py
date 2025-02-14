import streamlit as st
import sys
import numpy as np
from PIL import Image

sys.path.append('./manager')

from manager.manager import Manager 
manager = Manager()


def main():
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        segment_button = st.button('Segment Image')
        with col1:
            st.write("Original Image")
            st.image(uploaded_file, caption='Uploaded Image.', use_container_width=True)
        with col2:
            st.write("Result")
            if segment_button:
                image = Image.open(uploaded_file)
                img_array = np.array(image)
                with st.spinner("Wait a second please.", show_time=True):
                    seg_img_dict = manager.segmentation(img_array)
                    mask=np.expand_dims(seg_img_dict[st.session_state.channel], axis=-1)
                st.image(mask*img_array, caption='Segmented Lession.', use_container_width=True)
        

if __name__ == "__main__":
    st.title("Skin Lesions Segmentation")
    st.write("This is a demo of the Skin Lesions Segmentation model.")
    st.write("Upload an image of a skin lesion.")
    st.session_state.channel = st.selectbox(
        label="Select in what channel to segment the image.",
        options=('X_channel', 'XoYoR_channel', 'XoYoZoR_channel', 'R_channel', 'B_channel'),
        index=4,
    )
    main()
    bottom_placeholder = st.empty()
    bottom_placeholder.markdown(
        """
            ---\n 
            The model developed for this project is based on the research presented in the article 
            <a href="https://www.researchgate.net/profile/Alauddin-Bhuiyan/publication/238737819_Skin_Lesion_Segmentation_Using_Color_Channel_Optimization_and_Clustering-based_Histogram_Thresholding/links/5d085bcb458515ea1a6d79e8/Skin-Lesion-Segmentation-Using-Color-Channel-Optimization-and-Clustering-based-Histogram-Thresholding.pdf"
            target="_blank">
            \"Skin Lesion Segmentation Using Color Channel Optimization and 
            Clustering-based Histogram Thresholding\"
            </a>
        """, 
    unsafe_allow_html=True)