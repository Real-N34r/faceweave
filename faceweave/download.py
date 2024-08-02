import os
import ssl
import subprocess
import urllib.request
from functools import lru_cache
from typing import List
from urllib.parse import urlparse

from tqdm import tqdm

from faceweave import state_manager, wording
from faceweave.common_helper import is_macos
from faceweave.filesystem import get_file_size, is_file, remove_file

if is_macos():
	ssl._create_default_https_context = ssl._create_unverified_context


def conditional_download(download_directory_path : str, urls : List[str]) -> None:
	for url in urls:
		download_file_path = os.path.join(download_directory_path, os.path.basename(urlparse(url).path))
		initial_size = get_file_size(download_file_path)
		download_size = get_download_size(url)
		if initial_size < download_size:
			with tqdm(total = download_size, initial = initial_size, desc = wording.get('downloading'), unit = 'B', unit_scale = True, unit_divisor = 1024, ascii = ' =', disable = state_manager.get_item('log_level') in [ 'warn', 'error' ]) as progress:
				subprocess.Popen([ 'curl', '--create-dirs', '--silent', '--insecure', '--location', '--continue-at', '-', '--output', download_file_path, url ])
				current_size = initial_size
				while current_size < download_size:
					if is_file(download_file_path):
						current_size = get_file_size(download_file_path)
						progress.update(current_size - progress.n)
		if download_size and not is_download_done(url, download_file_path):
			remove_file(download_file_path)
			conditional_download(download_directory_path, [ url ])


@lru_cache(maxsize = None)
def get_download_size(url : str) -> int:
	try:
		response = urllib.request.urlopen(url, timeout = 10)
		content_length = response.headers.get('Content-Length')
		return int(content_length)
	except (OSError, TypeError, ValueError):
		return 0


def is_download_done(url : str, file_path : str) -> bool:
	if is_file(file_path):
		return get_download_size(url) == get_file_size(file_path)
	return False
