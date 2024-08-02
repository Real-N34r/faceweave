from functools import lru_cache
from time import sleep
from typing import Optional

import cv2
import numpy
from tqdm import tqdm

from faceweave import process_manager, state_manager, wording
from faceweave.download import conditional_download
from faceweave.execution import create_inference_pool
from faceweave.filesystem import is_file, resolve_relative_path
from faceweave.thread_helper import conditional_thread_semaphore, thread_lock
from faceweave.typing import Fps, InferencePool, ModelOptions, ModelSet, VisionFrame
from faceweave.vision import count_video_frame_total, detect_video_fps, get_video_frame, read_image

INFERENCE_POOL : Optional[InferencePool] = None
MODEL_SET : ModelSet =\
{
	'open_nsfw':
	{
		'sources':
		{
			'content_analyser':
			{
				'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/open_nsfw.onnx',
				'path': resolve_relative_path('../.assets/models/open_nsfw.onnx')
			}
		}
	}
}
PROBABILITY_LIMIT = 0.80
RATE_LIMIT = 10
STREAM_COUNTER = 0


def get_inference_pool() -> InferencePool:
	global INFERENCE_POOL

	with thread_lock():
		while process_manager.is_checking():
			sleep(0.5)
		if INFERENCE_POOL is None:
			model_sources = get_model_options().get('sources')
			INFERENCE_POOL = create_inference_pool(model_sources, state_manager.get_item('execution_device_id'), state_manager.get_item('execution_providers'))
		return INFERENCE_POOL


def clear_inference_pool() -> None:
	global INFERENCE_POOL

	INFERENCE_POOL = None


def get_model_options() -> ModelOptions:
	return MODEL_SET.get('open_nsfw')


def pre_check() -> bool:
	download_directory_path = resolve_relative_path('../.assets/models')
	model_sources = get_model_options().get('sources')
	model_urls = [ model_sources.get(model_source).get('url') for model_source in model_sources.keys() ]
	model_paths = [ model_sources.get(model_source).get('path') for model_source in model_sources.keys() ]

	if not state_manager.get_item('skip_download'):
		process_manager.check()
		conditional_download(download_directory_path, model_urls)
		process_manager.end()
	return all(is_file(model_path) for model_path in model_paths)


def analyse_stream(vision_frame : VisionFrame, video_fps : Fps) -> bool:
	global STREAM_COUNTER

	STREAM_COUNTER = STREAM_COUNTER + 1
	if STREAM_COUNTER % int(video_fps) == 0:
		return analyse_frame(vision_frame)
	return False


def analyse_frame(vision_frame : VisionFrame) -> bool:
	return False


def prepare_frame(vision_frame : VisionFrame) -> VisionFrame:
	vision_frame = cv2.resize(vision_frame, (224, 224)).astype(numpy.float32)
	vision_frame -= numpy.array([ 104, 117, 123 ]).astype(numpy.float32)
	vision_frame = numpy.expand_dims(vision_frame, axis = 0)
	return vision_frame


@lru_cache(maxsize = None)
def analyse_image(image_path : str) -> bool:
	return False


@lru_cache(maxsize = None)
def analyse_video(video_path : str, start_frame : int, end_frame : int) -> bool:
	return False
