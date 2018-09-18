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
              指明当前的控制系统是否是重新启动+
              
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
        dayDelta = (endDate-startDate).days
        for k in range(dayDelta//self.interval):
            if len(splitStages) == 0:
                splitStages.append((startDate,startDate+datetime.timedelta(self.interval)))
            else:
                startSplit = splitStages[-1][1]
                splitStages.append((startSplit,startSplit+datetime.timedelta(self.interval)))
        lastSplitStart = splitStages[-1][1] - datetime.timedelta(np.random.randint(low=self.interval//4,high=self.interval//2))
        splitStages.append((lastSplitStart,endDate))
        return splitStages
    
    def geneTestData(self,stageData,testLength):
        startDate = stageData.index[-1] + datetime.timedelta(0,60)
        endDate = stageData.index[-1] + datetime.timedelta(0,60*testLength)
        testDataFrame = self.dataReader(startDate,endDate).readData()
        testDataFrame = testDataFrame.fillna(method="ffill").fillna(method="bfill")
        return testDataFrame[self.featureColumns].as_matrix(),testDataFrame[self.targetColumns].as_matrix()
    
    def genePsudoData(self,stageData,weather,GasDis):
        GasDis = np.array(GasDis)
        feature = stageData[self.featureColumns].fillna(method='ffill').fillna(method='bfill').as_matrix()
        PsudoData = np.array([feature[-1]]*self.pLength)
        PsudoDatas = []
        if len(weather) > 1:
            PsudoData[:,self.weatherIndex] = weather
        
        for gasAmount in range(self.downGasAmount,self.upGasAmount,100):
            tempPsudoData = copy.deepcopy(PsudoData)
            tempPsudoData[:,self.gasIndex] = np.array([GasDis]*len(tempPsudoData))*gasAmount
            PsudoDatas.append(tempPsudoData)
            
        return PsudoDatas
    
    def main(self):
        # 如果是重启模型的话，就需要读取历史数据对模型重新训练
        if self.reStartModel:
            # 首先将原有的正则化文件和模型参数文件删除
            self.removeOriginalFiles(self.ModelStoreFiles)
            # 对已有的历史数据进行分组
            splitStage = self.splitTrainData()
            # 根据每一段时期的训练数据来训练模型
            for k,dates in enumerate(splitStage):
                SD,ED = dates
                print('正在加载历史数据（%s -- %s）'%(SD,ED))
                stageData = self.dataReader(SD,ED).readData()
                print('该阶段的历史数据加载结束')
                print('开始初始化模型')
                self.model = self.ModelClass(stageData,self.featureColumns,
                                    self.targetColumns,logDir=self.logDir)
                print('对模型进行训练')
                diff = self.model.trainModel(trainQuota=1,ecorr=1,controlStage=False)
        else:
            diff = 0.0
            
        print('++++++++++++++++运营阶段++++++++++++++')
        
        # 初始化必要的参数
        # 学习误差率的初始值为1
        ecorr = 1
        # 如果预测结果没有通过事前检测，可以重新训练的次数
        retryNum = 3
        # 记录控制周期
        controlLoops = 1
        # 多锅炉之间的总燃气的初始分配方案
        GasDis = self.initGasDis
        for t in range(100):
            print('开始第%d次系统控制'%controlLoops)
            currentTime = self.Clock.now()
            startTime = currentTime - datetime.timedelta(np.random.randint(9,15))
            print('对当前阶段历史数据进行加载（%s -- %s）'%(startTime,currentTime))
            stageData = self.dataReader(startTime,currentTime).readData()
            self.model = self.ModelClass(stageData,self.featureColumns,self.targetColumns,
                                logDir = self.logDir)
            print(u'对模型进行训练')
            diff = self.model.trainModel(0.5,ecorr,True)
            if self.testLabel:
                print('对模型当前的预测结果进行检验')
                testFeature,testTarget = self.geneTestData(stageData,60*48)
                predictTarget = self.model.predictModel(testFeature,self.pLength,
                                            self.testLabel,60*48)
                figure = plt.figure()
                ax = figure.add_subplot(111)
                ax.plot(predictTarget.ravel(),c='blue',lw=2,alpha=0.6,label="Prediction")
                ax.plot(testTarget.ravel(),c='red',lw=1.5,alpha=0.7,label='Reality')
                ax.legend(loc='best')
                figure.savefig('当前模型的预测结果（%s）.png'%currentTime,dpi=300)
                
            # 结合虚拟数据进行预测
            print('正在产生虚拟数据集')
            PsudoWeather = self.genePsudoWeather(stageData[self.featureColumns],
                                        self.weatherIndex,self.pLength)
            PsudoDatas = self.genePsudoData(stageData,PsudoWeather,GasDis)
            PredictGroups = []
            print('对虚拟数据集进行预测')
            for k,PsudoData in enumerate(PsudoDatas):
                print('.',end=' ')
                predictGroup = self.model.predictModel(PsudoData,self.pLength,predictIndex=k)
                PredictGroups.append(predictGroup)
            print()
            predictGroups = np.vstack(PredictGroups)
            # 根据预测偏置误差调整预测结果组
            predictGroups = predictGroups - diff
            if self.InnerChecker(predictGroups) < 200 or retryNum == 0:
                retryNum = 3
                print('正在计算最恰当的目标回水温度')
                targetBT,stableBT = self.geneTargetBT(stageData)
                print('目标回水温度：%.2f，稳定回水温度：%.2f'%(targetBT,stableBT))
                pursueBT = self.stableWeight * stableBT + \
                             self.purcheWeight * max(min(targetBT,stableBT+2),stableBT-2)
                print('根据权值分配，最合适的追踪回水温度为：%.2f'%pursueBT)
                pursueGas,stableGas = self.geneValidGas(predictGroups,pursueBT,stableBT)
                print('追踪回水温度所需要的燃气量为：%d，保持当前的回水温度所需要的燃气量为：%d'%(pursueGas,stableGas))
                savePath = os.path.join(self.logDir,'第%d个控制周期预测.txt'%controlLoops)
                np.savetxt(savePath,predictGroups,fmt='%.2f')
                
                endTime = currentTime + datetime.timedelta(0,60*self.pLength)
                while True:
                    cc = self.Clock.now()
                    if (endTime - cc).seconds < 20 or cc > endTime:
                        break
                    self.Clock.sleep(10)
                print
                print('对模型进行事后检验')
                SD = currentTime + datetime.timedelta(0,60)
                ED = endTime
                newData = self.dataReader(SD,ED).readData()
                newData = newData.fillna(method='ffill').fillna(method='bfill')
                GasDis,totalCorr = self.OutterChecker(predictGroups,newData,
                                        self.downGasAmount,self.upGasAmount,
                                        self.logDir,controlLoops)
                ecorr = 1 - totalCorr
                controlLoops += 1
            else:
                print('未通过事前检验，需要重新训练')
                retryNum -= 1
                continue
            
    def InnerChecker(self,predictGroups):
        """
            该方法要确保预测集中高燃气量对应高回水温度
        """
        diffs = []
        for k in range(predictGroups.shape[0]-1):
            i = predictGroups[k]
            j = predictGroups[k+1]
            diff = j - i
            diffs.append(diff)
        diffs = np.array(diffs)
        return sum((diffs < -0.03).ravel())
    
    def geneTargetBT(self,dataFrame):
        dataFrame = dataFrame.fillna(method='ffill').fillna(method='bfill')
        # 读取当前的时间和当前的室外温度
        currentTime = dataFrame.index[-1]
        currentHour = currentTime.hour
        currentOutTemp = dataFrame[u'气象站室外温度'].as_matrix().ravel()[-1]
        
        outT2BT_Data = []
        index = dataFrame.index
        outT = dataFrame[u'气象站室外温度'].as_matrix().ravel()
        BT = dataFrame[self.targetColumns].as_matrix().ravel()
        for k,i in enumerate(index):
            outT2BT_Data.append([i.hour,outT[k],BT[k]])
            
        def specificHourMap(hour):
            def func(x,a,b):
                return a*x + b
            
            specificHourData = []
            for item in outT2BT_Data:
                if item[0] == hour:
                    specificHourData.append([item[1],item[2]])
            specificHourData = np.array(specificHourData)
            paras,COV = curve_fit(func,specificHourData[:,0],specificHourData[:,1],p0=(0,0))
            print("参数估计时的协方差矩阵：")
            print(COV)
            validFunc = lambda x: paras[0] * x + paras[1]
            return validFunc
        
        MapFunc = specificHourMap(int(currentHour+(self.pLength//60)/2.0)%24)
        return MapFunc(currentOutTemp),BT[-1]
    
    def geneValidGas(self,predictGroups,pursueBT,stableBT):
        down = self.downGasAmount
        minPursueIndex = np.argmin(np.abs(np.mean(predictGroups[:,-3:],axis=1) - pursueBT))
        pursueGas = down + minPursueIndex * 100
        minStableIndex = np.argmin(np.abs(np.mean(predictGroups[:,-3:],axis=1) - stableBT))
        stableGas = down + minStableIndex * 100
        return pursueGas,stableGas

    
        
        
        