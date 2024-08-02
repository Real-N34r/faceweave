from typing import List, Optional, Tuple

import gradio

from faceweave import state_manager, wording
from faceweave.processors import choices as processors_choices
from faceweave.processors.core import load_processor_module
from faceweave.processors.typing import FaceEditorModel
from faceweave.uis.core import get_ui_component, register_ui_component

FACE_EDITOR_MODEL_DROPDOWN : Optional[gradio.Dropdown] = None
FACE_EDITOR_EYE_OPEN_RATIO_SLIDER : Optional[gradio.Slider] = None
FACE_EDITOR_EYE_OPEN_FACTOR_SLIDER : Optional[gradio.Slider] = None
FACE_EDITOR_LIP_OPEN_RATIO_SLIDER : Optional[gradio.Slider] = None
FACE_EDITOR_LIP_OPEN_FACTOR_SLIDER : Optional[gradio.Slider] = None


def render() -> None:
	global FACE_EDITOR_MODEL_DROPDOWN
	global FACE_EDITOR_EYE_OPEN_RATIO_SLIDER
	global FACE_EDITOR_EYE_OPEN_FACTOR_SLIDER
	global FACE_EDITOR_LIP_OPEN_RATIO_SLIDER
	global FACE_EDITOR_LIP_OPEN_FACTOR_SLIDER

	FACE_EDITOR_MODEL_DROPDOWN = gradio.Dropdown(
		label = wording.get('uis.face_editor_model_dropdown'),
		choices = processors_choices.face_editor_models,
		value = state_manager.get_item('face_editor_model'),
		visible = 'face_editor' in state_manager.get_item('processors')
	)
	FACE_EDITOR_EYE_OPEN_RATIO_SLIDER = gradio.Slider(
		label = wording.get('uis.face_editor_eye_open_ratio_slider'),
		value = state_manager.get_item('face_editor_eye_open_ratio'),
		step = processors_choices.face_editor_eye_open_ratio_range[1] - processors_choices.face_editor_eye_open_ratio_range[0],
		minimum = processors_choices.face_editor_eye_open_ratio_range[0],
		maximum = processors_choices.face_editor_eye_open_ratio_range[-1],
		visible = 'face_editor' in state_manager.get_item('processors'),
	)
	FACE_EDITOR_EYE_OPEN_FACTOR_SLIDER = gradio.Slider(
		label = wording.get('uis.face_editor_eye_open_factor_slider'),
		value = state_manager.get_item('face_editor_eye_open_factor'),
		step = processors_choices.face_editor_eye_open_factor_range[1] - processors_choices.face_editor_eye_open_factor_range[0],
		minimum = processors_choices.face_editor_eye_open_factor_range[0],
		maximum = processors_choices.face_editor_eye_open_factor_range[-1],
		visible = 'face_editor' in state_manager.get_item('processors'),
	)
	FACE_EDITOR_LIP_OPEN_RATIO_SLIDER = gradio.Slider(
		label = wording.get('uis.face_editor_lip_open_ratio_slider'),
		value = state_manager.get_item('face_editor_lip_open_ratio'),
		step = processors_choices.face_editor_lip_open_ratio_range[1] - processors_choices.face_editor_lip_open_ratio_range[0],
		minimum = processors_choices.face_editor_lip_open_ratio_range[0],
		maximum = processors_choices.face_editor_lip_open_ratio_range[-1],
		visible = 'face_editor' in state_manager.get_item('processors'),
	)
	FACE_EDITOR_LIP_OPEN_FACTOR_SLIDER = gradio.Slider(
		label = wording.get('uis.face_editor_lip_open_factor_slider'),
		value = state_manager.get_item('face_editor_lip_open_factor'),
		step = processors_choices.face_editor_lip_open_factor_range[1] - processors_choices.face_editor_lip_open_factor_range[0],
		minimum = processors_choices.face_editor_lip_open_factor_range[0],
		maximum = processors_choices.face_editor_lip_open_factor_range[-1],
		visible = 'face_editor' in state_manager.get_item('processors'),
	)
	register_ui_component('face_editor_model_dropdown', FACE_EDITOR_MODEL_DROPDOWN)
	register_ui_component('face_editor_eye_open_ratio_slider', FACE_EDITOR_EYE_OPEN_RATIO_SLIDER)
	register_ui_component('face_editor_eye_open_factor_slider', FACE_EDITOR_EYE_OPEN_FACTOR_SLIDER)
	register_ui_component('face_editor_lip_open_ratio_slider', FACE_EDITOR_LIP_OPEN_RATIO_SLIDER)
	register_ui_component('face_editor_lip_open_factor_slider', FACE_EDITOR_LIP_OPEN_FACTOR_SLIDER)


def listen() -> None:
	FACE_EDITOR_MODEL_DROPDOWN.change(update_face_editor_model, inputs = FACE_EDITOR_MODEL_DROPDOWN, outputs = FACE_EDITOR_MODEL_DROPDOWN)
	FACE_EDITOR_EYE_OPEN_RATIO_SLIDER.release(update_face_editor_eye_open_ratio, inputs = FACE_EDITOR_EYE_OPEN_RATIO_SLIDER)
	FACE_EDITOR_EYE_OPEN_FACTOR_SLIDER.release(update_face_editor_eye_open_factor, inputs = FACE_EDITOR_EYE_OPEN_FACTOR_SLIDER)
	FACE_EDITOR_LIP_OPEN_RATIO_SLIDER.release(update_face_editor_lip_open_ratio, inputs = FACE_EDITOR_LIP_OPEN_RATIO_SLIDER)
	FACE_EDITOR_LIP_OPEN_FACTOR_SLIDER.release(update_face_editor_lip_open_factor, inputs = FACE_EDITOR_LIP_OPEN_FACTOR_SLIDER)

	processors_checkbox_group = get_ui_component('processors_checkbox_group')
	if processors_checkbox_group:
		processors_checkbox_group.change(remote_update, inputs = processors_checkbox_group, outputs = [ FACE_EDITOR_MODEL_DROPDOWN, FACE_EDITOR_EYE_OPEN_RATIO_SLIDER, FACE_EDITOR_EYE_OPEN_FACTOR_SLIDER, FACE_EDITOR_LIP_OPEN_RATIO_SLIDER, FACE_EDITOR_LIP_OPEN_FACTOR_SLIDER ])


def remote_update(processors : List[str]) -> Tuple[gradio.Dropdown, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider]:
	has_face_editor = 'face_editor' in processors
	return gradio.Dropdown(visible = has_face_editor), gradio.Slider(visible = has_face_editor), gradio.Slider(visible = has_face_editor), gradio.Slider(visible = has_face_editor), gradio.Slider(visible = has_face_editor)


def update_face_editor_model(face_editor_model : FaceEditorModel) -> gradio.Dropdown:
	face_editor_module = load_processor_module('face_editor')
	face_editor_module.clear_inference_pool()
	state_manager.set_item('face_editor_model', face_editor_model)

	if face_editor_module.pre_check():
		return gradio.Dropdown(value = state_manager.get_item('face_editor_model'))
	return gradio.Dropdown()


def update_face_editor_eye_open_ratio(face_editor_eye_open_ratio : float) -> None:
	state_manager.set_item('face_editor_eye_open_ratio', face_editor_eye_open_ratio)


def update_face_editor_eye_open_factor(face_editor_eye_open_factor : float) -> None:
	state_manager.set_item('face_editor_eye_open_factor', int(face_editor_eye_open_factor))


def update_face_editor_lip_open_ratio(face_editor_lip_open_ratio : float) -> None:
	state_manager.set_item('face_editor_lip_open_ratio', face_editor_lip_open_ratio)


def update_face_editor_lip_open_factor(face_editor_lip_open_factor : float) -> None:
	state_manager.set_item('face_editor_lip_open_factor', int(face_editor_lip_open_factor))
