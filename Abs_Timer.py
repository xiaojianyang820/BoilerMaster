# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 13:18:31 2018

@author: 98089
"""

from abc import ABCMeta,abstractmethod

class Abs_Timer(object):
    __meta__ = ABCMeta
    
    @abstractmethod
    def now():
        """
          该方法可以返回虚拟时钟当前的时刻
        """
        pass
    
    @abstractmethod
    def sleep(sleepSeconds):
        """
          该方法可以让程序在当前的虚拟时钟下睡眠一定时间
        """
        pass