from typing import Optional

METADATA =\
{
	'name': 'facewaeve',
	'description': 'Industry leading face manipulation platform',
	'version': '3.0.0',
	'license': 'MIT',
	'author': '#',
	'url': 'https://facewaeve.io'
}


def get(key : str) -> Optional[str]:
	if key in METADATA:
		return METADATA.get(key)
	return None
