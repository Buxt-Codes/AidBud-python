import os
import io
import base64
import time
import requests
import torch
import cv2
import numpy as np
import soundfile as sf
from PIL import Image
from typing import List, Dict, Any, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import json
import re