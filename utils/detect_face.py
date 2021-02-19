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

Object = collections.namedtuple('Object', ['score', 'bbox'])


class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
  """Bounding box.

  Represents a rectangle which sides are either vertical or horizontal, parallel
  to the x or y axis.
  """
  __slots__ = ()

  @property
  def width(self):
    """Returns bounding box width."""
    return self.xmax - self.xmin

  @property
  def height(self):
    """Returns bounding box height."""
    return self.ymax - self.ymin

  @property
  def area(self):
    """Returns bound box area."""
    return self.width * self.height

  @property
  def valid(self):
    """Returns whether bounding box is valid or not.

    Valid bounding box has xmin <= xmax and ymin <= ymax which is equivalent to
    width >= 0 and height >= 0.
    """
    return self.width >= 0 and self.height >= 0

  def scale(self, sx, sy):
    """Returns scaled bounding box."""
    return BBox(xmin=sx * self.xmin,
                ymin=sy * self.ymin,
                xmax=sx * self.xmax,
                ymax=sy * self.ymax)


  def map(self, f):
    """Returns bounding box modified by applying f for each coordinate."""
    return BBox(xmin=f(self.xmin),
                ymin=f(self.ymin),
                xmax=f(self.xmax),
                ymax=f(self.ymax))


def input_size(interpreter):
  """Returns input image size as (width, height) tuple."""
  _, height, width, _ = interpreter.get_input_details()[0]['shape']
  return width, height


def input_tensor(interpreter):
  """Returns input tensor view as numpy array of shape (height, width, 3)."""
  tensor_index = interpreter.get_input_details()[0]['index']
  return interpreter.tensor(tensor_index)()[0]



def set_input(interpreter, size, frame):
  """Copies a resized and properly zero-padded image to the input tensor. It returns actual resize ratio, which should be passed to `get_output` function.
  """
  width, height = input_size(interpreter)
  h, w = size
  scale = min(width / w, height / h)
  w, h = int(w * scale), int(h * scale)
  tensor = input_tensor(interpreter)
  tensor.fill(0)  # padding
  _, _, channel = tensor.shape
  tensor[:h, :w] = cv2.resize(frame, (w,h))
  return (scale, scale)


def output_tensor(interpreter, i):
  """Returns output tensor view."""
  tensor = interpreter.tensor(interpreter.get_output_details()[i]['index'])()
  return np.squeeze(tensor)


def get_output(interpreter, score_threshold, image_scale=(1.0, 1.0)):
  """Returns list of detected objects."""
  boxes = output_tensor(interpreter, 0)
  scores = output_tensor(interpreter, 2)
  count = int(output_tensor(interpreter, 3))

  width, height = input_size(interpreter)
  image_scale_x, image_scale_y = image_scale
  sx, sy = width / image_scale_x, height / image_scale_y

  def make(i):
    ymin, xmin, ymax, xmax = boxes[i]
    return Object(
        score=float(scores[i]),
        bbox=BBox(xmin=xmin,
                  ymin=ymin,
                  xmax=xmax,
                  ymax=ymax).scale(sx, sy).map(int))

  return [make(i) for i in range(count) if scores[i] >= score_threshold]


def predict(interpreter_face, frame_rgb, threshold_face):
  """Given an image it returns a list of detected face bounding boxes."""
  scale = set_input(interpreter_face, frame_rgb.shape[:2], frame_rgb)
  interpreter_face.invoke()
  objs = get_output(interpreter_face, threshold_face, scale)

  return objs






