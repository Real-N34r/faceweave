from typing import List, Optional, Tuple

import gradio

from faceweave import state_manager, wording
from faceweave.processors import choices as processors_choices
from faceweave.processors.core import load_processor_module
from faceweave.processors.typing import FrameColorizerModel
from faceweave.uis.core import get_ui_component, register_ui_component

FRAME_COLORIZER_MODEL_DROPDOWN : Optional[gradio.Dropdown] = None
FRAME_COLORIZER_BLEND_SLIDER : Optional[gradio.Slider] = None
FRAME_COLORIZER_SIZE_DROPDOWN : Optional[gradio.Dropdown] = None


def render() -> None:
	global FRAME_COLORIZER_MODEL_DROPDOWN
	global FRAME_COLORIZER_BLEND_SLIDER
	global FRAME_COLORIZER_SIZE_DROPDOWN

	FRAME_COLORIZER_MODEL_DROPDOWN = gradio.Dropdown(
		label = wording.get('uis.frame_colorizer_model_dropdown'),
		choices = processors_choices.frame_colorizer_models,
		value = state_manager.get_item('frame_colorizer_model'),
		visible = 'frame_colorizer' in state_manager.get_item('processors')
	)
	FRAME_COLORIZER_BLEND_SLIDER = gradio.Slider(
		label = wording.get('uis.frame_colorizer_blend_slider'),
		value = state_manager.get_item('frame_colorizer_blend'),
		step = processors_choices.frame_colorizer_blend_range[1] - processors_choices.frame_colorizer_blend_range[0],
		minimum = processors_choices.frame_colorizer_blend_range[0],
		maximum = processors_choices.frame_colorizer_blend_range[-1],
		visible = 'frame_colorizer' in state_manager.get_item('processors')
	)
	FRAME_COLORIZER_SIZE_DROPDOWN = gradio.Dropdown(
		label = wording.get('uis.frame_colorizer_size_dropdown'),
		choices = processors_choices.frame_colorizer_sizes,
		value = state_manager.get_item('frame_colorizer_size'),
		visible = 'frame_colorizer' in state_manager.get_item('processors')
	)
	register_ui_component('frame_colorizer_model_dropdown', FRAME_COLORIZER_MODEL_DROPDOWN)
	register_ui_component('frame_colorizer_blend_slider', FRAME_COLORIZER_BLEND_SLIDER)
	register_ui_component('frame_colorizer_size_dropdown', FRAME_COLORIZER_SIZE_DROPDOWN)


def listen() -> None:
	FRAME_COLORIZER_MODEL_DROPDOWN.change(update_frame_colorizer_model, inputs = FRAME_COLORIZER_MODEL_DROPDOWN, outputs = FRAME_COLORIZER_MODEL_DROPDOWN)
	FRAME_COLORIZER_BLEND_SLIDER.release(update_frame_colorizer_blend, inputs = FRAME_COLORIZER_BLEND_SLIDER)
	FRAME_COLORIZER_SIZE_DROPDOWN.change(update_frame_colorizer_size, inputs = FRAME_COLORIZER_SIZE_DROPDOWN)

	processors_checkbox_group = get_ui_component('processors_checkbox_group')
	if processors_checkbox_group:
		processors_checkbox_group.change(remote_update, inputs = processors_checkbox_group, outputs = [ FRAME_COLORIZER_MODEL_DROPDOWN, FRAME_COLORIZER_BLEND_SLIDER, FRAME_COLORIZER_SIZE_DROPDOWN ])


def remote_update(processors : List[str]) -> Tuple[gradio.Dropdown, gradio.Slider, gradio.Dropdown]:
	has_frame_colorizer = 'frame_colorizer' in processors
	return gradio.Dropdown(visible = has_frame_colorizer), gradio.Slider(visible = has_frame_colorizer), gradio.Dropdown(visible = has_frame_colorizer)


def update_frame_colorizer_model(frame_colorizer_model : FrameColorizerModel) -> gradio.Dropdown:
	frame_colorizer_module = load_processor_module('frame_colorizer')
	frame_colorizer_module.clear_inference_pool()
	state_manager.set_item('frame_colorizer_model', frame_colorizer_model)

	if frame_colorizer_module.pre_check():
		return gradio.Dropdown(value = state_manager.get_item('frame_colorizer_model'))
	return gradio.Dropdown()


def update_frame_colorizer_blend(frame_colorizer_blend : float) -> None:
	state_manager.set_item('frame_colorizer_blend', int(frame_colorizer_blend))


def update_frame_colorizer_size(frame_colorizer_size : str) -> None:
	state_manager.set_item('frame_colorizer_size', frame_colorizer_size)
