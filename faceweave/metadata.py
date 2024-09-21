from typing import Optional

METADATA =\
{
	'name': 'faceweave',
	'description': 'Industry leading face manipulation platform',
	'version': '3.0.0',
	'license': 'MIT',
	'author': '#',
	'url': 'https://faceweave.io'
}


def get(key : str) -> Optional[str]:
	if key in METADATA:
		return METADATA.get(key)
	return None
