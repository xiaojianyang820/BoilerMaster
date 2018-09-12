# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 13:52:09 2018

@author: 98089
"""

from abc import ABCMeta, abstractmethod

class Abs_ModelClass(object):
    __meta__ = ABCMeta
    
    # 后续必须实现的类属性
    featureColumns = []
    targetColumns = []
    weatherIndex = []
    gasIndex = []
    ModelStoreFiles = []
    
    def __init__(self,stageData,FeatureColumns,TargetColumns,logDir):
        """
            stageData:DataFrame
              该阶段已有的历史数据
            FeatureColumns:List
              训练集用到的字段名
            TargetColumns:List
              目标集用到的字段名
            logDir:路径字符串
              日志或者缓存文件夹
        """
        pass
    
    @abstractmethod
    def trainModel(self,trainQuota,ecorr,controlStage):
        '''
            trainQuota:0-1浮点型
              训练强度
            ecorr:0-1浮点型
              上一轮次模型的误差率
            controlStage:Boolean型
              当前模型所处的阶段，False指向预训练阶段，True代表运行阶段
        '''
        # 将拟合结果和真实结果的差异diff返回回去
        diff = 0
        return diff
    
    @abstractmethod    
    def predictModel(self,predictX,pLength,testLabel=False,testLength=60*48,predictIndex=0):
        """
            predictX:Array
              用于预测的特征数据集矩阵
            pLength:Int
              每一个控制周期的预测长度
            testLabel:Boolean
              是否在该轮次中调用真实历史数据对模型的预测效果进行评估，
            testLength:Int
              如果需要对预测效果进行评估，那么所抽取的真实历史数据的长度为多少
        """
        # 将预测的结果序列返回回去
        predictY = []
        return predictY