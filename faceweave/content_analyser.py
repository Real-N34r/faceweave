from functools import lru_cache
import cv2
import numpy
from tqdm import tqdm

from faceweave import inference_manager, state_manager, wording
from faceweave.download import conditional_download_hashes, conditional_download_sources
from faceweave.filesystem import resolve_relative_path
from faceweave.thread_helper import conditional_thread_semaphore
from faceweave.typing import Fps, InferencePool, ModelOptions, ModelSet, VisionFrame
from faceweave.vision import count_video_frame_total, detect_video_fps, get_video_frame, read_image

# Model settings are retained but won't be used
MODEL_SET : ModelSet = {
    'open_nsfw': {
        'hashes': {
            'content_analyser': {
                'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/open_nsfw.hash',
                'path': resolve_relative_path('../.assets/models/open_nsfw.hash')
            }
        },
        'sources': {
            'content_analyser': {
                'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/open_nsfw.onnx',
                'path': resolve_relative_path('../.assets/models/open_nsfw.onnx')
            }
        },
        'size': (224, 224),
        'mean': [104, 117, 123]
    }
}

PROBABILITY_LIMIT = 0.80
RATE_LIMIT = 10
STREAM_COUNTER = 0

# Skip inference pool, since NSFW detection is disabled
def get_inference_pool() -> InferencePool:
    return None

def clear_inference_pool() -> None:
    pass

def get_model_options() -> ModelOptions:
    return MODEL_SET.get('open_nsfw')

def pre_check() -> bool:
    # Skip NSFW model checks
    return True

# NSFW detection disabled by returning False always
def analyse_stream(vision_frame: VisionFrame, video_fps: Fps) -> bool:
    global STREAM_COUNTER
    STREAM_COUNTER += 1
    if STREAM_COUNTER % int(video_fps) == 0:
        return analyse_frame(vision_frame)
    return False

# NSFW detection disabled by returning False always
def analyse_frame(vision_frame: VisionFrame) -> bool:
    # Skip NSFW detection and return False always
    return False

# Skip forward pass as NSFW detection is off
def forward(vision_frame: VisionFrame) -> float:
    return 0.0

# Keeping the frame preparation function in case it's used elsewhere
def prepare_frame(vision_frame: VisionFrame) -> VisionFrame:
    model_size = get_model_options().get('size')
    model_mean = get_model_options().get('mean')
    vision_frame = cv2.resize(vision_frame, model_size).astype(numpy.float32)
    vision_frame -= numpy.array(model_mean).astype(numpy.float32)
    vision_frame = numpy.expand_dims(vision_frame, axis=0)
    return vision_frame

@lru_cache(maxsize=None)
def analyse_image(image_path: str) -> bool:
    frame = read_image(image_path)
    return analyse_frame(frame)

@lru_cache(maxsize=None)
def analyse_video(video_path: str, start_frame: int, end_frame: int) -> bool:
    video_frame_total = count_video_frame_total(video_path)
    video_fps = detect_video_fps(video_path)
    frame_range = range(start_frame or 0, end_frame or video_frame_total)
    rate = 0.0
    counter = 0

    with tqdm(total=len(frame_range), desc=wording.get('analysing'), unit='frame', ascii=' =', disable=state_manager.get_item('log_level') in ['warn', 'error']) as progress:
        for frame_number in frame_range:
            if frame_number % int(video_fps) == 0:
                frame = get_video_frame(video_path, frame_number)
                # Since NSFW detection is off, this will always return False
                if analyse_frame(frame):
                    counter += 1
            rate = counter * int(video_fps) / len(frame_range) * 100
            progress.update()
            progress.set_postfix(rate=rate)
    
    # Rate limit check is now bypassed as NSFW detection is disabled
    return rate > RATE_LIMIT
