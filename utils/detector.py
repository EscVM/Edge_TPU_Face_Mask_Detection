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
import argparse
import time
from utils import detect_face
from utils import detect_mask
import tflite_runtime.interpreter as tflite
import platform
import cv2
from threading import Thread
import os


class Detector():
  """Class for live camera detection"""
  def __init__(self, cpu_face, cpu_mask, models_path, threshold_face, camera, threshold_mask):
    self.cpu_face = cpu_face
    self.cpu_mask = cpu_mask
    # path FaceNet
    if self.cpu_face:
      self.MODEL_PATH_FACE = os.path.join(models_path, 'ssd_mobilenet_v2_face_quant_postprocess.tflite')
    else: 
      self.MODEL_PATH_FACE = os.path.join(models_path, 'ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite')
    # path MaskNet
    if self.cpu_mask:
      self.MODEL_PATH_FACE_MASK = os.path.join(models_path, 'mobilenet_v2_mask_classification.tflite')
    else:
      self.MODEL_PATH_FACE_MASK = os.path.join(models_path, 'mobilenet_v2_mask_classification_edgetpu.tflite')

    self.threshold_face = threshold_face
    self.frame_bytes = None
    self.camera = camera
    self.threshold_mask = threshold_mask
    self.mask_labels = ['No Mask', 'Mask']

  def make_interpreter(self, model_file, cpu):
    """Create an interpreter delegating on the tpu or cpu"""
      # set some parameters 
    EDGETPU_SHARED_LIB = {
      'Linux': 'libedgetpu.so.1',
      'Darwin': 'libedgetpu.1.dylib',
      'Windows': 'edgetpu.dll'
    }[platform.system()]

    model_file, *device = model_file.split('@')
    if not cpu:
      interpreter =  tflite.Interpreter(
          model_path=model_file,
          experimental_delegates=[
              tflite.load_delegate(EDGETPU_SHARED_LIB,
                                   {'device': device[0]} if device else {})])
    else:
      interpreter =  tflite.Interpreter(
          model_path=model_file)
    return interpreter

  def draw_objects(self, frame, objs, y_mask_pred, fps):
    """Draws the bounding box for each object."""
    for i, obj in enumerate(objs):
        color = (255,255,255)  # white color if mask not classified yet
        bbox = obj.bbox
        # mask detection
        if len(y_mask_pred) != 0:
          y_pred = y_mask_pred[i]
          label = self.mask_labels[y_pred > self.threshold_mask]

          if label == self.mask_labels[0]:
            color = (0,0,255) # b g r, red color if mask not detected
          else:
            color = (0,255,0)

          cv2.rectangle(frame, (int(bbox[0] - 2), int(bbox[1] - 45)), (int(bbox[2] + 2), int(bbox[1])), color, -1)
          cv2.putText(frame,
                  '{} {:.1%}'.format(label, y_pred),
                  (int(bbox.xmin + 5), int(bbox.ymin - 10)),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  2.5 * ((bbox.xmax - bbox.xmin)/frame.shape[0]),
                  (255,255,255),
                  2,
                  cv2.LINE_AA)
          
        
        cv2.rectangle(frame, 
                    (int(bbox.xmin), int(bbox.ymin)), 
                    (int(bbox.xmax), int(bbox.ymax)),
                   color, 
                   3)

    cv2.putText(frame, 'FPS:{:.4}'.format(fps), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)


  def start(self):
    """Main loop function."""
    # initialize coral accelerator
    interpreter_face = self.make_interpreter(self.MODEL_PATH_FACE, self.cpu_face)
    interpreter_face.allocate_tensors()

    # initialize face mask classificer
    interpreter_mask = self.make_interpreter(self.MODEL_PATH_FACE_MASK, self.cpu_mask)
    interpreter_mask.allocate_tensors()


    # define some variables
    camera = cv2.VideoCapture(self.camera)
    cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)

    y_mask_pred = []


    # start loop
    while(True):
      # get opencv data
      ret, frame = camera.read()

      t0 = time.clock()

      frame_rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)

      # faces detection
      objs = detect_face.predict(interpreter_face, frame_rgb, self.threshold_face)

      # mask detection
      if len(objs) != 0:
        try:
          y_mask_pred = detect_mask.predict(interpreter_mask, frame_rgb, objs)
        except:
          y_mask_pred = []
        
      t1 = time.clock()
      
      self.draw_objects(frame, objs, y_mask_pred, (1/(t1-t0)))
      
      cv2.imshow('Camera', frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        # When everything done, release the capture
        camera.release()
        cv2.destroyAllWindows()
        break

  


class Detector_Thread(Thread, Detector):
  """Mutli-Thread class for live camera detection."""
  def __init__(self, cpu_face, cpu_mask, models_path, threshold_face, camera, threshold_mask):
    Thread.__init__(self)
    Detector.__init__(self, cpu_face, cpu_mask, models_path, threshold_face, camera, threshold_mask)

  def run(self):
    """Main loop function."""
    # initialize coral accelerator
    interpreter_face = self.make_interpreter(self.MODEL_PATH_FACE, self.cpu_face)
    interpreter_face.allocate_tensors()

    # initialize face mask classificer
    interpreter_mask = self.make_interpreter(self.MODEL_PATH_FACE_MASK, self.cpu_mask)
    interpreter_mask.allocate_tensors()


    # define some variables
    camera = cv2.VideoCapture(self.camera)

    y_mask_pred = []


    # start loop
    while(True):
      # get opencv data
      ret, frame = camera.read()

      t0 = time.clock()

      frame_rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)

      # faces detection
      objs = detect_face.predict(interpreter_face, frame_rgb, self.threshold_face)

      # mask detection
      if len(objs) != 0:
        try:
          y_mask_pred = detect_mask.predict(interpreter_mask, frame_rgb, objs)
        except:
          y_mask_pred = []
        
      t1 = time.clock()

      
      self.draw_objects(frame, objs, y_mask_pred, (1/(t1-t0)))
      
      self.frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()

  def get_frame(self):
    """Return a frame for the server"""
    return self.frame_bytes
