'''
主程序框架
'''

from utils import Asymetric
from data_for_asy import X_train, X_test
import sys
import os

sys.path.append('/mnt/datadisk2/aglv/foraglv/DataDaily.cpython-38-x86_64-linux-gnu.so')
sys.path.append('')

obj = Asymetric(X_train)
