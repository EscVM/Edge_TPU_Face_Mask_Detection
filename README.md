[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<h1 align="center"> ~ Face Mask Detection with Edge TPU ~ </h1>

It's time for a new, time-waster project🥳!! Actually, this little repository is a little bit less pointless than other [ones](https://github.com/EscVM/RPS_with_Edge_TPU) and could be useful for someone. 
I exploited the power of Edge TPUs to build a face mask detector for Covid-19 (I really had to do something for Covid-19 😂).

Anyway, the code is straightforward: there's a network trained to recognize faces in an image and another one that detects the presence of the mask. I've found the first network [here](https://coral.ai/models/), and I trained the second one with this little awful [dataset](https://drive.google.com/drive/folders/1XDte2DL2Mf_hw4NsmGst7QtYoU7sMBVG) (I leave a Colab notebook to train different networks). Everything is optimized for Edge TPU inference, but you can run all the code on a CPU changing [configurations](https://github.com/EscVM/Edge_TPU_Face_Mask_Detection/blob/main/config.json). I only used opencv-python and the TensorFlow-Lite interpreter. With a couple of faces, it runs around 50 fps, as you can see in the example below.

<p align="center">
  <img width="600" height="338" src="media/demo.gif">
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
   
If you want a mini server version.

   ```bash
   python3 main_server.py
   ```

Once started, search on your browser [](localhost:8080). Login with the username and password 'admin'/'admin' (what else👀?).


# 3.0 Train and Optimized a New Mask Detector

Coming soon...
