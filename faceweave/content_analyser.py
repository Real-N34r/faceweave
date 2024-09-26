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

# Model setup (NSFW filtering disabled, keeping the structure intact)
MODEL_SET: ModelSet = {}

PROBABILITY_LIMIT = 0.80  # Original probability limit, now unused
RATE_LIMIT = 10           # Original rate limit, now unused
STREAM_COUNTER = 0         # Keeps track of stream frame counting


def get_inference_pool() -> InferencePool:
    """NSFW Model is disabled, so inference pool is not used."""
    return {}


def clear_inference_pool() -> None:
    """Clear any models, though none are loaded since NSFW filtering is disabled."""
    inference_manager.clear_inference_pool(__name__)


def pre_check() -> bool:
    """Pre-check for models, returning True since no models are needed."""
    return True


def analyse_stream(vision_frame: VisionFrame, video_fps: Fps) -> bool:
    """Process each frame in a stream, ensuring that invalid frames are skipped."""
    global STREAM_COUNTER
    STREAM_COUNTER += 1
    if STREAM_COUNTER % int(video_fps) == 0:
        if vision_frame is None:
            print(f"Warning: Frame is None at counter {STREAM_COUNTER}. Skipping analysis.")
            return False
        return analyse_frame(vision_frame)
    return False


def analyse_frame(vision_frame: VisionFrame) -> bool:
    """Perform analysis on a single frame (NSFW filtering disabled)."""
    if vision_frame is None:
        print("Warning: Received NoneType frame. Skipping frame analysis.")
        return False

    # Processing without the NSFW model
    vision_frame = prepare_frame(vision_frame)
    print("Skipping NSFW filtering, proceeding without content analysis.")
    return False  # Always returning False since no filtering is applied


def forward(vision_frame: VisionFrame) -> float:
    """Forward pass (now unused, as NSFW filtering is disabled)."""
    raise NotImplementedError("Forward function is disabled as NSFW filtering is turned off.")


def prepare_frame(vision_frame: VisionFrame) -> VisionFrame:
    """Prepare the frame (resize and normalize) for processing."""
    if vision_frame is None:
        raise ValueError("Cannot prepare a NoneType frame. Skipping.")

    model_size = (224, 224)  # Default size, though unused since NSFW filtering is off
    model_mean = [104, 117, 123]

    # Resize the frame and normalize it
    vision_frame = cv2.resize(vision_frame, model_size).astype(numpy.float32)
    vision_frame -= numpy.array(model_mean).astype(numpy.float32)
    vision_frame = numpy.expand_dims(vision_frame, axis=0)
    return vision_frame


@lru_cache(maxsize=None)
def analyse_image(image_path: str) -> bool:
    """Perform analysis on an image, skipping NSFW filtering."""
    frame = read_image(image_path)
    if frame is None:
        print(f"Warning: Could not read image from {image_path}.")
        return False
    return analyse_frame(frame)


@lru_cache(maxsize=None)
def analyse_video(video_path: str, start_frame: int, end_frame: int) -> bool:
    """Perform analysis on a video (NSFW filtering disabled)."""
    video_frame_total = count_video_frame_total(video_path)
    video_fps = detect_video_fps(video_path)
    frame_range = range(start_frame or 0, end_frame or video_frame_total)
    rate = 0.0
    counter = 0

    with tqdm(total=len(frame_range), desc=wording.get('analysing'), unit='frame',
              ascii=' =', disable=state_manager.get_item('log_level') in ['warn', 'error']) as progress:
        for frame_number in frame_range:
            if frame_number % int(video_fps) == 0:
                frame = get_video_frame(video_path, frame_number)
                if frame is None:
                    print(f"Warning: Frame {frame_number} is None. Skipping.")
                    continue
                if analyse_frame(frame):
                    counter += 1
            rate = counter * int(video_fps) / len(frame_range) * 100
            progress.update()
            progress.set_postfix(rate=rate)
    return rate > RATE_LIMIT


def get_video_frame(video_path, frame_number):
    """Retrieve a video frame, ensuring NoneType is handled."""
    video_capture = cv2.VideoCapture(video_path)
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = video_capture.read()
    video_capture.release()

    if not success or frame is None:
        print(f"Warning: Could not read frame {frame_number} from {video_path}.")
        return None

    return frame
