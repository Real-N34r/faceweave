from typing import Optional

METADATA =\
{
	'name': 'faceweave',
	'description': 'Next generation face swapper and enhancer',
	'version': 'NEXT',
	'license': 'MIT',
	'author': 'Henry Ruhs',
	'url': 'https://faceweave.io'
}


def get(key : str) -> Optional[str]:
	if key in METADATA:
		return METADATA.get(key)
	return None
