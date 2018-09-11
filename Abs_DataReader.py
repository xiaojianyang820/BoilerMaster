# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 14:50:11 2018

@author: 98089
"""

from abc import ABCMeta,abstractmethod

class Abs_DataReader(object):
    __meta__ = ABCMeta
    
    @abstractmethod
    def __init__(self,startDate,endDate):
        pass
    
    @abstractmethod
    def readData(self):
        # 返回对应的历史数据
        pass