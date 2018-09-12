# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 17:13:29 2018

@author: 98089
"""
from abc import abstractmethod

@abstractmethod
def OutterChecker(predictGroups,newData,downGasAmount,upGasAmount,logDir,controlLoops):
    
    # 返回值应该依次为下一控制周期设备的启停方案，上一控制周期的总评分，
    #return GasDis,score