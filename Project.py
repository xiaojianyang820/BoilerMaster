# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 15:02:03 2018

@author: 98089
"""
from Timer import Timer
from DataReader import XK
from CNNModel import CNNModel
from main import MainModel
from OutterChecker import OutterChecker

def genePsudoWeather(stageData,weatherIndex,pLength):
    return [-1]

def toLog(level,part,content):
    with open("logs/log.csv",'a') as f:
    	f.write('(%s)[%s]:%s\n'%(part,level,content))
    	print('(%s)[%s]:%s\n'%(part,level,content))


if __name__ == "__main__":
    localClock = Timer('2018-02-17 00:00:00',20)
    modelClass = CNNModel
    model = MainModel(False,'2017-12-01 00:00:00',localClock,30,modelClass,
                      [1],True,genePsudoWeather,XK,"./logs",120,800,4000,toLog,OutterChecker)
    model.main()