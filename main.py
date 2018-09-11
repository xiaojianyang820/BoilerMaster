# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 12:06:52 2018

@author: 98089
"""

import numpy as np
import os,datetime,copy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tools import tools_DataTimeTrans


class MainModel(object):
    
    def __init__(self,reStartModel,startDate,interval,Clock,ModelClass,initGasDis,
                 testLabel,genePsudoWeather,dataReader,logDir,pLength,downGasAmount,
                 upGasAmount,toLog,OutterChecker):
        """
            对控制系统的初始化参数进行设定
            
            reStartModel:Boolean
              指明当前的控制系统是否是重新启动
              
            startDate:Str
              如果当前的控制系统是重新启动的话，那么加载历史数据的起始时刻是哪里，
              格式为"%Y-%m-%d %H:%M:%S"
              
            interval:Int
              对历史数据进行学习时，其分割的大小
              
            Clock:Timer类对象
              该对象为模型提供虚拟的系统时钟，该对象有两个方法，now()方法提供当前虚拟系统
              的时刻，sleep(Int)方法让程序在当前的虚拟时钟下睡眠一段时间
            
            ModelClass:CNNModel类
              该类主要实现了机器学习模型，该类的初始化方法有四个参数，依次是阶段历史数据
              DataFrame，训练字段名List，目标字段名List，日志存储文件夹logDir；
              trainModel方法，参数是trainQuota，0-1的浮点型，用来指示模型应有的训练强度，ecorr，0-1浮点型，
              用来指示上一次预测模型的误差率，如果是首次训练，那么该值为1，controlStage，
              Boolean型，来指明模型当前所处的状态，预训练状态还是运营状态，该方法对模型
              的参数进行学习，同时提供一个返回值，diff，来指明当前模型的预测值和实际值之间
              的差距，以对预测的系统性误差进行补充；predictModel方法，三个参数，第一是
              testFeature,matrix类型，为了进行预测的构造的虚拟特征数据集矩阵，第二是
              pLength，Int类型，预测序列的长度，第三是testLabel，声明本次预测是否需要
              读取真实的历史数据进行预测效果评估，本参数只适用于回测模型阶段，第四个参数
              是testLength，指明在testLabel为True时，回测模型预测的时间序列长度，该
              方法返回预测的结果序列
              
            initGasDis:List
              初始化的供暖设备启停方案
              
            testLabel:Boolean
              指示模型在预测过程中是否对预测效果进行检验，只用于回测阶段
              
            genePsudoWeather:函数
              该函数有三个参数，第一个是当前阶段的历史数据，DataFrame；第二个是数据集中关于
              天气的字段位置，List；第三个是预测长度，Int
              
            dataReader:DataReader类
              该对象主要负责数据源和主控制程序之间的数据读取工作，该对象有两个必要的方法，
              第一个是初始化方法，接受两个参数，分别是数据的起始时间和结束时间；第二个是
              readData()方法，无参数，将指定阶段的历史数据（DataFrame）返回
              
            logDir:路径字符串
              指明日志文件和缓存文件的存储文件夹
              
            pLength:Int
              一个控制周期的长度
              
            downGasAmount:Int
              测试燃气量的下限
              
            upGasAmount:Int
              测试燃气量的上限
              
            toLog:函数
              接受三个参数，第一个是part，哪一个系统所发出的日志，第二个是level，即日志
              的级别，第三个是content，即日志的内容
              
            OutterChecker:函数
              接受六个参数，第一个是预测结果组，第二个是预测的起始时刻，第三个是预测的结束时刻
              第四个是测试的燃气下限，第五个是测试的燃气上限，第六个是日志文件的存储位置
        """
        self.reStartModel = reStartModel
        self.startDate = startDate
        self.Clock = Clock
        self.interval = interval
        self.ModelClass = ModelClass
        self.featureColumns = ModelClass.featureColumns
        self.targetColumns = ModelClass.targetColumns
        self.weatherIndex = ModelClass.weatherIndex
        self.gasIndex = ModelClass.gasIndex
        self.ModelStoreFiles = ModelClass.ModelStoreFiles
        self.toLog = toLog
        self.OutterChecker = OutterChecker
        
        # 对模型类中的必要类属性定义进行检查
        if len(self.featureColumns) == 0:
            raise ValueError('模型类中未对其所引用的特征列名进行定义')
        if len(self.targetColumns) == 0:
            raise ValueError('模型类中未对其所引用的目标列名进行定义')
        if len(self.weatherIndex) == 0:
            raise ValueError('模型类中未指明天气相关数据的索引')
        if len(self.gasIndex) == 0:
            raise ValueError('模型类中未指明燃气相关数据的索引')
        if len(self.ModelStoreFiles) == 0:
            self.toLog('警告','机器学习系统','似乎没有指定存储模型参数的文件列表')
        
        self.initGasDis = initGasDis
        self.testLabel = testLabel
        self.genePsudoWeather = genePsudoWeather
        self.dataReader = dataReader
        self.logDir = logDir
        self.pLength = pLength
        self.upGasAmount = upGasAmount
        self.downGasAmount = downGasAmount
        # 原有水温和目标水温之间的分配比例
        self.stableWeight = 0.5
        self.purcheWeight = 1-self.stableWeight
        
    def removeOriginalFiles(self,files):
        """
            如果重新启动模型，就对涉及到的原有的参数文件进行删除
        """
        for fileName in files:
            filePath = os.path.join(self.logDir,fileName)
            if os.path.exists(filePath):
                os.remove(filePath)
            else:
                self.toLog('警告','流程控制系统','文件（%s）不存在，请检查输入是否正确'%fileName)
    
    def splitTrainData(self):
        startDate = tools_DataTimeTrans(self.startDate)
        endDate = self.Clock.now()
        splitStages = []
        
        
        