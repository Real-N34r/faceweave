from typing import Dict, Optional, Any, List
from types import ModuleType
import os
import importlib
import sys
import gradio
from faceweave.uis.themes.theme import Applio
import faceweave.globals
from faceweave.uis import overrides
from faceweave import metadata, logger, wording
from faceweave.uis.typing import Component, ComponentName
from faceweave.filesystem import resolve_relative_path

os.environ['GRADIO_ANALYTICS_ENABLED'] = '0'

gradio.processing_utils.encode_array_to_base64 = overrides.encode_array_to_base64
gradio.processing_utils.encode_pil_to_base64 = overrides.encode_pil_to_base64

UI_COMPONENTS: Dict[ComponentName, Component] = {}
UI_LAYOUT_MODULES : List[ModuleType] = []
UI_LAYOUT_METHODS =\
[
	'pre_check',
	'pre_render',
	'render',
	'listen',
	'run'
]


def load_ui_layout_module(ui_layout : str) -> Any:
	try:
		ui_layout_module = importlib.import_module('faceweave.uis.layouts.' + ui_layout)
		for method_name in UI_LAYOUT_METHODS:
			if not hasattr(ui_layout_module, method_name):
				raise NotImplementedError
	except ModuleNotFoundError as exception:
		logger.error(wording.get('ui_layout_not_loaded').format(ui_layout = ui_layout), __name__.upper())
		logger.debug(exception.msg, __name__.upper())
		sys.exit(1)
	except NotImplementedError:
		logger.error(wording.get('ui_layout_not_implemented').format(ui_layout = ui_layout), __name__.upper())
		sys.exit(1)
	return ui_layout_module


def get_ui_layouts_modules(ui_layouts : List[str]) -> List[ModuleType]:
	global UI_LAYOUT_MODULES

	if not UI_LAYOUT_MODULES:
		for ui_layout in ui_layouts:
			ui_layout_module = load_ui_layout_module(ui_layout)
			UI_LAYOUT_MODULES.append(ui_layout_module)
	return UI_LAYOUT_MODULES


def get_ui_component(component_name : ComponentName) -> Optional[Component]:
	if component_name in UI_COMPONENTS:
		return UI_COMPONENTS[component_name]
	return None


def get_ui_components(component_names : List[ComponentName]) -> Optional[List[Component]]:
	ui_components = []

	for component_name in component_names:
		component = get_ui_component(component_name)
		if component:
			ui_components.append(component)
	return ui_components


def register_ui_component(component_name : ComponentName, component: Component) -> None:
	UI_COMPONENTS[component_name] = component


def launch() -> None:
	ui_layouts_total = len(faceweave.globals.ui_layouts)
	with gradio.Blocks(theme = Applio(),  title = metadata.get('name') + ' ' + metadata.get('version')) as ui:
		for ui_layout in faceweave.globals.ui_layouts:
			ui_layout_module = load_ui_layout_module(ui_layout)
			if ui_layout_module.pre_render():
				if ui_layouts_total > 1:
					with gradio.Tab(ui_layout):
						ui_layout_module.render()
						ui_layout_module.listen()
				else:
					ui_layout_module.render()
					ui_layout_module.listen()

	for ui_layout in faceweave.globals.ui_layouts:
			ui_layout_module = load_ui_layout_module(ui_layout)
			ui_layout_module.run(ui)
