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
import os
from utils.tools import load_config
from utils.detector import Detector
import cv2

# import parameters from configuration file
config = load_config(config_path='config.json')

cpu_face = config['cpu_face']
cpu_mask = config['cpu_mask']
threshold_face = config['threshold_face']
camera = config['camera']
threshold_mask = config['threshold_mask']
models_path = config['models_path']


if __name__ == '__main__':
	detector = Detector(cpu_face, cpu_mask, models_path, threshold_face, camera, threshold_mask)
	detector.start()
