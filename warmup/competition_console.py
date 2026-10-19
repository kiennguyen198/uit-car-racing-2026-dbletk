"""Competition terminal output and internal diagnostic stream."""
import os
import sys
from pathlib import Path

_terminal=None
_stderr=None
_log=None

def begin():
    global _terminal,_stderr,_log
    folder=Path(__file__).resolve().parent/'logs'
    folder.mkdir(exist_ok=True)
    sys.stdout.flush();sys.stderr.flush()
    _terminal=os.dup(1);_stderr=os.dup(2)
    _log=os.open(folder/'terminal_internal.log',os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
    os.dup2(_log,1);os.dup2(_log,2)

def message(text):
    os.write(_terminal,(text+'\n').encode('utf-8'))

def ready():
    message('SẴN SÀNG')

def failure(error):
    text=str(error)
    if 'Connection failed' in text or 'Connection lost' in text:
        message('Không kết nối được simulator ở cổng 11000.')
    elif 'CUDA unavailable' in text:
        message('Không có GPU CUDA. Kiểm tra NVIDIA driver và cấu hình Docker.')
    else:
        message('Chương trình gặp lỗi. Chi tiết được lưu trong /workspace/logs/terminal_internal.log.')

def finish():
    if _terminal is None:return
    sys.stdout.flush();sys.stderr.flush()
    os.dup2(_terminal,1);os.dup2(_stderr,2)
    os.close(_terminal);os.close(_stderr);os.close(_log)
