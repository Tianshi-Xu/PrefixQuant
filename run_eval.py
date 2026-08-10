import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_TRUST_REMOTE_CODE'] = '1'

import socket
_orig_connect = socket.socket.connect
def _blocked_connect(self, address, *args, **kwargs):
    host, port = address
    if port in (443, 80) or 'huggingface' in str(host) or 'github' in str(host):
        raise ConnectionError(f'Blocked: {address}')
    return _orig_connect(self, address, *args, **kwargs)
socket.socket.connect = _blocked_connect

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table

socket.socket.connect = _orig_connect

from eval import main
main()
