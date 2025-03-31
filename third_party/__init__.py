import sys
import os

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_path, 'Matcha-TTS'))
sys.path.append(os.path.join(file_path))  # 添加当前目录到路径
# 导入CosyVoice和相关工具
from .cosyvoice.cli.cosyvoice import CosyVoice2
from .cosyvoice.utils.file_utils import load_wav
from .cosyvoice.utils.common import set_all_random_seed

__all__ = ['CosyVoice2', 'load_wav', 'set_all_random_seed']
