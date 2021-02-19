# Copyright 2021 Vittorio Mazzia. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import collections
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt


def input_tensor(interpreter):
    """Returns input tensor view as numpy array of shape (height, width, 3)."""
    tensor_index = interpreter.get_input_details()[0]['index']
    return interpreter.tensor(tensor_index)()[0]


def set_input(interpreter, data):
    """Copies data to input tensor."""
    scale, zero_point = interpreter.get_input_details()[0]['quantization']
    input_tensor(interpreter)[:, :] = (data / scale) + zero_point
    

def output_tensor(interpreter):
    """Returns dequantized output tensor."""
    output_details = interpreter.get_output_details()[0]
    output_data = np.squeeze(interpreter.tensor(output_details['index'])())
    scale, zero_point = output_details['quantization']
    return np.abs(scale * (output_data - zero_point) - 1) * 1.25

def input_size(interpreter):
  """Returns input image size as (width, height) tuple."""
  _, height, width, _ = interpreter.get_input_details()[0]['shape']
  return width, height



def predict(interpreter, frame_rgb, objs):
  """Given a list of box objects, it predicts for each box if it's present a mask. It returns an array of predictions."""
  y_pred = []
  width, height = input_size(interpreter)
  
  for obj in objs:
    bbox = obj.bbox
    x_face = frame_rgb[int(bbox.ymin):int(bbox.ymax), int(bbox.xmin):int(bbox.xmax)]
    x_face_resized = cv2.resize(x_face, (height, width))
    x_face_resized = preprocess_input(x_face_resized)

    set_input(interpreter, x_face_resized)

    # invoke interpreter
    interpreter.invoke()

    y_pred.append(output_tensor(interpreter))
        
  return np.array(y_pred)


