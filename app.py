import streamlit as st
import numpy as np
import cv2
import os
from inference import main
from dataclasses import dataclass
from typing import Optional


class InferenceArguments:
    def __init__(self, inputs, output) -> None:
        self.model_arch_name = "srresnet_x4"
        self.compile_state = False
        self.model_weights_path = "/Users/viethungnguyen/SRGAN-PyTorch/results/pretrained_models/SRResNet_x4-SRGAN_ImageNet.pth.tar"
        self.inputs = inputs
        self.output = output
        self.device = "cpu"
        self.half = False


if __name__ == "__main__":
    st.title("Image Super Resolution")
    uploaded_file = st.file_uploader("Choose an image...")
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        file_name = uploaded_file.name
        input_file_path = os.path.join(
            "/Users/viethungnguyen/SRGAN-PyTorch/figure", file_name
        )
        with open(input_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        output_file_path = f"/Users/viethungnguyen/SRGAN-PyTorch/figure/sr_{file_name}"
        arguments = InferenceArguments(inputs=input_file_path, output=output_file_path)
        main(args=arguments)

        col1, col2 = st.columns(2)
        with col1:
            st.header("Low Resolution")
            st.image(input_file_path)

        with col2:
            st.header("Super Resolution")
            st.image(output_file_path)
