from typing import List, Optional

import gradio

from faceweave import content_analyser, face_analyser, face_masker, state_manager, voice_extractor, wording
from faceweave.execution import get_execution_provider_choices
from faceweave.processors.core import clear_processors_modules
from faceweave.typing import ExecutionProviderKey

EXECUTION_PROVIDERS_CHECKBOX_GROUP : Optional[gradio.CheckboxGroup] = None


def render() -> None:
	global EXECUTION_PROVIDERS_CHECKBOX_GROUP

	EXECUTION_PROVIDERS_CHECKBOX_GROUP = gradio.CheckboxGroup(
		label = wording.get('uis.execution_providers_checkbox_group'),
		choices = get_execution_provider_choices(),
		value = state_manager.get_item('execution_providers')
	)


def listen() -> None:
	EXECUTION_PROVIDERS_CHECKBOX_GROUP.change(update_execution_providers, inputs = EXECUTION_PROVIDERS_CHECKBOX_GROUP, outputs = EXECUTION_PROVIDERS_CHECKBOX_GROUP)


def update_execution_providers(execution_providers : List[ExecutionProviderKey]) -> gradio.CheckboxGroup:
	clear_processors_modules()
	content_analyser.clear_inference_pool()
	face_analyser.clear_inference_pool()
	face_masker.clear_inference_pool()
	voice_extractor.clear_inference_pool()
	execution_providers = execution_providers or get_execution_provider_choices()
	state_manager.set_item('execution_providers', execution_providers)
	return gradio.CheckboxGroup(value = state_manager.get_item('execution_providers'))
