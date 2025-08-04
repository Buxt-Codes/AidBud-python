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
from ...config import Config

try:
    from transformers import AutoProcessor, AutoModelForImageTextToText
except ImportError:
    print("Hugging Face `transformers` is not installed. Please run `pip install transformers`.")
    AutoProcessor = None
    AutoModelForImageTextToText = None

try:
    import timm
except ImportError:
    print("`timm` library is not installed. Please run `pip install timm`.")
    timm = None
    
try:
    import soundfile as sf
except ImportError:
    print("`soundfile` library is not installed. Please run `pip install soundfile`.")
    sf = None

try:
    from moviepy import VideoFileClip
except ImportError:
    print("`moviepy` library is not installed. Please run `pip install moviepy`.")
    VideoFileClip = None

try:
    import cv2
except ImportError:
    print("`cv2` library is not installed. Please run `pip install cv2`.")
    cv2 = None


class LLM:
    def __init__(self, model_id: str = "google/gemma-3n-E4B-it", config: Config = Config()):
        if AutoProcessor is None or AutoModelForImageTextToText is None:
            raise RuntimeError("Hugging Face `transformers` library not found. Cannot initialize.")

        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model '{self.model_id}' on device: {self.device}...")

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        print("Model loaded successfully.")
        
        self.fps = config.llm["fps"]
        self.max_workers = config.llm["num_workers"] if config.llm["num_workers"] is not None else os.cpu_count()

        self.image_processing = True
        self.video_processing = True
        self.video_audio_processing = True
        self.audio_processing = True

        if cv2 is None:
            print("OpenCV is not available. Cannot process video.")
            self.video_processing = False
        
        if VideoFileClip is None:
            print("moviepy library is not available. Cannot extract audio from videos.")
            self.video_audio_processing = False
        
        if sf is None:
            print("Soundfile library is not available. Cannot process audio.")
            self.audio_processing = False

    def _prepare_prompt(self, prompt: str, images: List[Image.Image] = None, audios: List[np.ndarray] = None) -> List[Dict[str, Any]]:
        images = images if images is not None else []
        audios = audios if audios is not None else []

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        if images:
            messages[0]["content"].extend([{"type": "image", "image": image} for image in images])
        
        if audios:
            messages[0]["content"].extend([{"type": "audio", "audio": audio} for audio in audios])

        return messages
    
    def _prepare_image(self, path: str) -> Image.Image:
        try:
            if path.startswith("http://") or path.startswith("https://"):
                img = Image.open(requests.get(path, stream=True).raw)
            else:
                img = Image.open(path)
            return img
        except Exception as e:
            print(f"Warning: Could not process image at {path}. Error: {e}.")
            return None
    
    def _prepare_video(self, video_path: str) -> Tuple[List[Image.Image], Union[np.ndarray, None]]:
        images = []
        audio = None
        
        if video_path.startswith("http://") or video_path.startswith("https://"):
            try:
                print(f"Downloading video from URL: {video_path}")
                response = requests.get(video_path, stream=True)
                response.raise_for_status() 
            
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        temp_file.write(chunk)
                    temp_file_path = temp_file.name
                    print(f"Video downloaded to temporary file: {temp_file_path}")

                video_reader = cv2.VideoCapture(temp_file_path)
                if not video_reader.isOpened():
                    print(f"Warning: Could not open video file at {temp_file_path}. Skipping.")
                    os.remove(temp_file_path)
                    return [], None

                base_fps = video_reader.get(cv2.CAP_PROP_FPS)
                frame_interval = int(round(base_fps / self.fps))
                frame_count = 0
                while True:
                    success, frame = video_reader.read()
                    if not success:
                        break
                    if frame_count % frame_interval == 0:
                        try:
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            images.append(Image.fromarray(rgb_frame))
                        except Exception as e:
                            print(f"Warning: Could not process frame {frame_count}. Error: {e}")
                    frame_count += 1
                    if frame_count > video_reader.get(cv2.CAP_PROP_FRAME_COUNT):
                        break
                video_reader.release()

                if self.video_audio_processing:
                    try:
                        video_clip = VideoFileClip(temp_file_path)
                        audio_clip = video_clip.audio
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_temp_file:
                            audio_temp_path = audio_temp_file.name
                        audio_clip.write_audiofile(audio_temp_path, codec='pcm_s16le', fps=44100, logger=None)
                        audio_data, _ = sf.read(audio_temp_path, dtype='float32')
                        if audio_data.ndim > 1:
                            audio_data = audio_data.mean(axis=1) 
                        audio = audio_data
                        video_clip.close()
                        os.remove(audio_temp_path)
                    except Exception as e:
                        print(f"Warning: Could not extract audio from video {temp_file_path}. Error: {e}")

                os.remove(temp_file_path)

            except Exception as e:
                print(f"Warning: Failed to download or process video from URL {video_path}. Error: {e}")
        else:
            video_reader = cv2.VideoCapture(video_path)
            if not video_reader.isOpened():
                print(f"Warning: Could not open video file at {video_path}. Skipping.")
                return [], None

            base_fps = video_reader.get(cv2.CAP_PROP_FPS)
            frame_interval = int(round(base_fps / self.fps))
            frame_count = 0
            while True:
                success, frame = video_reader.read()
                if not success:
                    break
                if frame_count % frame_interval == 0:
                    try:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        images.append(Image.fromarray(rgb_frame))
                    except Exception as e:
                        print(f"Warning: Could not process frame {frame_count}. Error: {e}")
                frame_count += 1
                if frame_count > video_reader.get(cv2.CAP_PROP_FRAME_COUNT):
                    break
            video_reader.release()
            
            if self.video_audio_processing:
                try:
                    video_clip = VideoFileClip(video_path)
                    audio_clip = video_clip.audio
                    audio_buffer = io.BytesIO()
                    audio_clip.write_audiofile(audio_buffer, codec='pcm_s16le', fps=44100, logger=None)
                    audio_buffer.seek(0)
                    
                    audio_data, _ = sf.read(audio_buffer, dtype='float32')
                    if audio_data.ndim > 1:
                        audio_data = audio_data.mean(axis=1) 
                    audio = audio_data
                    
                    video_clip.close()
                except Exception as e:
                    print(f"Warning: Could not extract audio from video {video_path}. Error: {e}")
        
        return images, audio

    def _prepare_audio(self, path: str) -> np.ndarray:
        audio_data = None
        try:
            if path.startswith("http://") or path.startswith("https://"):
                audio_response = requests.get(path)
                audio_data, samplerate = sf.read(io.BytesIO(audio_response.content))
            else:
                audio_data, samplerate = sf.read(path)
            
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            return audio_data
        except Exception as e:
            print(f"Warning: Could not read audio file at {path}. Error: {e}")
            return None

    def generate(
        self, 
        prompt: str,
        image_paths: List[str] = None,
        video_paths: List[str] = None,
        audio_paths: List[str] = None
    ) -> str:
        image_paths = image_paths if image_paths is not None else []
        video_paths = video_paths if video_paths is not None else []
        audio_paths = audio_paths if audio_paths is not None else []
        images = []
        audios = []

        # with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
        #     if image_paths and self.image_processing:
        #         image_futures = [executor.submit(self._prepare_image, path) for path in image_paths]
        #         for future in as_completed(image_futures):
        #             img = future.result()
        #             if img:
        #                 images.append(img)
            
        #     if video_paths and self.video_processing:
        #         video_futures = [executor.submit(self._prepare_video, path) for path in video_paths]
        #         for future in as_completed(video_futures):
        #             video_frames, video_audio = future.result()
        #             if video_frames:
        #                 images.extend(video_frames)
        #             if video_audio:
        #                 audios.extend(video_audio)
            
        #     if audio_paths and self.audio_processing:
        #         audio_futures = [executor.submit(self._prepare_audio, path) for path in audio_paths]
        #         for future in as_completed(audio_futures):
        #             audio_data = future.result()
        #             if audio_data is not None:
        #                 audios.append(audio_data)

        if image_paths and self.image_processing:
            for path in image_paths:
                img = self._prepare_image(path)
                if img:
                    images.append(img)

        if video_paths and self.video_processing:
            for path in video_paths:
                video_frames, video_audio = self._prepare_video(path)
                if video_frames:
                    images.extend(video_frames)
                if video_audio is not None:
                    audios.append(video_audio)

        if audio_paths and self.audio_processing:
            for path in audio_paths:
                audio_data = self._prepare_audio(path)
                if audio_data is not None:
                    audios.append(audio_data)

        messages = self._prepare_prompt(prompt, images, audios)
        prompt_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors=False
        )

        inputs = self.processor(
            text=prompt_text,
            images=images if images else None,
            audio=audios if audios else None,
            return_tensors="pt"
        )
        
        model_dtype = next(self.model.parameters()).dtype
        inputs = {
            k: (
                v.to(self.model.device, dtype=model_dtype)
                if v.dtype in [torch.float16, torch.bfloat16, torch.float32]
                else v.to(self.model.device)
            )
            for k, v in inputs.items()
        }
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024
        )

        response_text = self.processor.batch_decode(outputs, skip_special_tokens=True)
        return response_text[0].strip()