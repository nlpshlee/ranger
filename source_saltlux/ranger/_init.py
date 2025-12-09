import os, sys


PATH = os.path.abspath(os.path.dirname(__file__))
#print(f"\ncurrent_path : {PATH}")

ROOT_DEPTH = 1
for i in range(ROOT_DEPTH) :
    PATH = os.path.dirname(PATH)
#print(f"\nroot_path : {PATH}, depth : {ROOT_DEPTH}")

# 최상위 경로 저장
sys.path.append(PATH)

# 전역 import 설정
from ranger.globals import *

