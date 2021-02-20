[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<h1 align="center"> ~ Face Mask Detection with Edge TPU ~ </h1>

Face mask detection on Edge TPU at more than 50 fps. The code is very straightforward: there's a network trained to recognize faces in an image and another one that detects the presence of the mask. The first network can be found [here](https://coral.ai/models/), and the second one has been trained with this little [dataset](https://drive.google.com/drive/folders/1XDte2DL2Mf_hw4NsmGst7QtYoU7sMBVG) (A Colab notebook is provided to train a new classifier on top of a different backbone). Everything is optimized for Edge TPU inference, but it's possible to run all the code on a CPU changing [configurations](https://github.com/EscVM/Edge_TPU_Face_Mask_Detection/blob/main/config.json). Only opencv-python and the TensorFlow-Lite interpreter are needed. As it's possible to see in the example below, it runs around 50 fps with a couple of faces with less than 3W! Enjoy 👨‍💻

<p align="center">
  <img width="600" height="340" src="media/demo.gif">
</p>

# 1.0 Getting Started
Clone this repository

   ```bash
   git clone https://github.com/EscVM/Edge_TPU_Face_Mask_Detection
   ```
## 1.1 Installations for the hosting device

Install on the hosting device the following libraries:

- [opencv-python](https://pypi.org/project/opencv-python/)
- [numpy](https://pypi.org/project/numpy/)
- [Flask](https://pypi.org/project/Flask/) (Optional)
- [TensorFlow Lite Interpreter](https://www.tensorflow.org/lite/guide/python) If you're using the Coral USB Accelerator with the Raspberry download ARM32.     

# 2.0 Run Face Mask Detector

   ```bash
   python3 main.py
   ```
   
Instead, if you want a mini server version, run the following command:

   ```bash
   python3 main_server.py
   ```

Once started, search on your browser [localhost:8080](http://localhost:8080). Login with the username and password 'admin'/'admin' (what👀?).


# 3.0 Train and Optimize a New Mask Detector
With the following notebook you can easily train a new classifier on top of whitchever backbone found [here](https://keras.io/api/applications/#densenet)(almost).
Once trained and converted you can place it in the [models](https://github.com/EscVM/Edge_TPU_Face_Mask_Detection/tree/main/models) folder. Rember to change paths in the [detector](https://github.com/EscVM/Edge_TPU_Face_Mask_Detection/blob/main/utils/detector.py) module.<br/><br/>
<a href="https://colab.research.google.com/drive/1kgEGysvTbL_1S7_X6pDwfVw7g28w8LnD?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
