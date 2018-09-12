#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 10:01:27 2018

@author: hadoop
"""

import pickle,os
from sklearn import preprocessing
import numpy as np
import torch
from torch import nn,optim
from torch.autograd import Variable
import pandas as pd
import datetime
from scipy import stats
import matplotlib.pyplot as plt

class CNNModel(object):
    
    featureColumns = [u"瞬时流量",u'气象站室外温度',u"气象站室外湿度",u"气象站室外风速",
                      u"回水压力",u'供水流量',u"燃气温度"]
    targetColumns = [u'一次回温度']
    weatherIndex = [1,2,3]
    gasIndex = [0]
    ModelStoreFiles = ['scores.txt','MainModelParas.z','CacheData.z','FScaler.z','TScaler.z']
    outChannel = 15
    step = 300
    hidden = 10
    upLag = 1500
    weight_decay = 0.001
    learningRate = 0.01
    lossCriterion = 'L1'
    
    def __init__(self,stageData,featureColumns,targetColumns,logDir):
        # 声明日志文件夹,特征组正则化文件，目标组正则化文件
        self.logDir = logDir
        self.FScalerFile = os.path.join(self.logDir,self.ModelStoreFiles[3])
        self.TScalerFile = os.path.join(self.logDir,self.ModelStoreFiles[4])
        # 存储子模型评分的文件
        self.scoresFile = self.ModelStoreFiles[0]
        # 存储主模型参数的文件
        self.MainModelParas = self.ModelStoreFiles[1]
        # 存储缓存数据的文件名
        self.CacheDataFile = self.ModelStoreFiles[2]
        # 加载历史数据集
        self.dataFrame = stageData
        # 检查是否有数据正则化函数
        self.haveScalerLabel = self.haveScaler()
        self.featureColumns = featureColumns
        self.targetColumns = targetColumns
        self.WDColumnsNames = u'气象站室外风向（东风）,气象站室外风向（西风）,气象站室外风向（南风）,气象站室外风向（北风）'
        # 初始化模型结构
        inChannel = len(featureColumns) + 1
        outChannel = self.outChannel
        step = self.step
        hidden = self.hidden
        output = len(targetColumns)
        upLag = self.upLag
        self.model = CNNMode_Kernal_2(inChannel,outChannel,step,hidden,upLag,
                                      output).cuda()
        if self.havPreTrainedRecord():
            print(u'模型正在加载历史结构参数')
            self.model.load_state_dict(torch.load(os.path.join(self.logDir,self.MainModelParas)))\
            
    def havPreTrainedRecord(self):
        return os.path.exists(os.path.join(self.logDir,self.MainModelParas))
        
        
    def haveScaler(self):
        FScalerPath = os.path.join(self.logDir,self.FScalerFile)
        TScalerPath = os.path.join(self.logDir,self.TScalerFile)
        if os.path.exists(FScalerPath) and os.path.exists(TScalerPath):
            with open(FScalerPath,'rb') as f:
                self.featureScaler = pickle.load(f)
            with open(TScalerPath,'rb') as f:
                self.targetScaler = pickle.load(f)
            return True
        else:
            return False
        
    def transDataFrame(self,dataFrame,feature,target,scalerLabel):
        hour = []
        for item in dataFrame.index:
            hour.append(item.hour)
        hour = np.array(hour).reshape(-1,1)
        
        if not self.haveScalerLabel:
            self.featureScaler = preprocessing.StandardScaler()
            self.targetScaler = preprocessing.StandardScaler()
            self.featureScaler.fit(feature)
            self.targetScaler.fit(target)
            with open(self.FScalerFile,'wb') as f:
                pickle.dump(self.featureScaler,f)
            with open(self.TScalerFile,'wb') as f:
                pickle.dump(self.targetScaler,f)
            self.haveScalerLabel = True
            
        if scalerLabel:
            feature = self.featureScaler.transform(feature)
            target = self.targetScaler.transform(target)
        feature = np.hstack([feature,hour])
        newFeature = []
        newTarget = []
        featureT = feature.T
        upLag = self.upLag
        for k in range(upLag,feature.shape[0]):
            newFeature.append(featureT[:,k-upLag:k])
            newTarget.append(target[k,0])
        return np.array(newFeature),np.array(newTarget)
    
    def isNull(self,Feature):
        validIndex = []
        for k,item in enumerate(Feature):
            if np.isnan(item.ravel()).sum() < 5:
                validIndex.append(k)
        return np.array(validIndex)
    
    def preprocess(self):
        self.dataFrame = self.dataFrame.fillna(method='ffill',limit=15).fillna(
                            method='bfill',limit=15)
        NoNull_dataFrame = self.dataFrame.fillna(method='ffill').fillna(method='bfill')
        NoNull_feature = NoNull_dataFrame[self.featureColumns].as_matrix()
        NoNull_target = NoNull_dataFrame[self.targetColumns].as_matrix()
        self.NoNull_feature = NoNull_feature
        self.NoNull_dataFrame = NoNull_dataFrame
        newFeature,newTarget = self.transDataFrame(NoNull_dataFrame,NoNull_feature,NoNull_target,True)
        feature = self.dataFrame[self.featureColumns].as_matrix()
        target = self.dataFrame[self.targetColumns].as_matrix()
        NullFeature,NullTarget = self.transDataFrame(self.dataFrame,feature,target,False)
        validIndex = self.isNull(NullFeature)
        print("由于数据缺失，抛弃掉的样本点数量：%d"%(len(NoNull_dataFrame) - len(validIndex),))
        weatherData = self.dataFrame[self.WDColumnsNames.split(',')].as_matrix()
        
        ClockData = [item.hour for item in self.dataFrame.index]
        ClockData = np.array(ClockData)
        
        return newFeature[validIndex],newTarget[validIndex],weatherData[validIndex],ClockData[validIndex]
            
    def covKernelLoss(self):
        criterion = nn.L1Loss()
        optimizer = optim.Adam(self.model.layer_10.parameters(),lr=0.02)
        realCovKernel = list(self.model.layer_10.parameters())[0]
        realCovKernelArray = realCovKernel.data.cpu().numpy().ravel()
        targetCovKernelArray = []
        plusOrMinus = abs(np.mean(realCovKernelArray))/np.mean(realCovKernelArray)
        for i in realCovKernelArray:
            if plusOrMinus * i < 0:
                targetCovKernelArray.append(0)
            else:
                targetCovKernelArray.append(i)
        
        temp = []
        for k,_ in enumerate(targetCovKernelArray):
            if k <= 5:
                temp.append(np.mean(targetCovKernelArray[0:5]))
            elif k >= len(targetCovKernelArray) - 5:
                temp.append(np.mean(targetCovKernelArray[len(targetCovKernelArray)-5:]))
            else:
                temp.append(np.mean(targetCovKernelArray[k-4:k+4]))
        targetCovKernelArray = np.array(temp)
        
        targetCovKernelArray = np.array(targetCovKernelArray,dtype=np.float32).reshape(1,1,-1)
        targetCovKernel = torch.from_numpy(targetCovKernelArray).cuda()
        V_targetCovKernel = Variable(targetCovKernel)
        loss = criterion(realCovKernel,V_targetCovKernel)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    def trainModel(self,trainQuota,ecorr,controlStage):
        trainEpoch = int(trainQuota*300)
        try:
            with open(os.path.join(self.logDir,self.scoresFile),'r') as f:
                corr = f.read().split(',')
                corr = np.array([float(i) for i in corr]) * ecorr * 3.3
                corr = np.array([min(1,i) for i in corr])
        except:
            print('上一轮次学习的效果缺失，如果是第一轮控制，则忽略该信息')
            corr = np.array([0.25,0.25,0.25,0.25])
        if not controlStage:
            return self.preTrain(trainEpoch,1)
        else:
            return self.controlTrain(trainEpoch,1-corr)
        
    def preTrain(self,totalEpoch,corr):
        if self.havPreTrainedRecord():
            print('模型正在加载历史结构参数')
            self.model.load_state_dict(torch.load(os.path.join(self.logDir,self.MainModelParas)))
            
        WD = self.weight_decay
        LR = self.learningRate * corr
        
        if self.lossCriterion == 'L1':
            criterion = nn.L1Loss()
        elif self.lossCriterion == 'L2':
            criterion = nn.MSELoss()
        else:
            print(u'需要指定误差函数类型')
            raise ValueError
        optimizer = optim.Adam(self.model.parameters(),lr=LR,weight_decay=WD)
        X,Y,weatherData,ClockData = self.preprocess()
        print('开始存储数据')
        cacheDatas = []
        for k,data in enumerate(zip(X,Y,weatherData,ClockData)):
            span = 11
            if k%span == 0:
                cacheData = []
                x,y,_,_ = data
                z = np.mean(weatherData[k-60:k],axis=0)
                c = np.mean(ClockData[k-60:k])
                x = x.ravel()
                cacheData.extend(x)
                cacheData.extend([y])
                cacheData.extend(z)
                cacheData.extend([c])
                cacheDatas.append(cacheData)
        cacheDatas = np.array(cacheDatas)
        if os.path.exists(os.path.join(self.logDir,self.CacheDataFile)):
            with open(os.path.join(self.logDir,self.CacheDataFile),'rb') as f:
                originalData = pickle.load(f)
            cacheDatas = np.vstack([originalData,cacheDatas])
        with open(os.path.join(self.logDir,self.CacheDataFile),'wb') as f:
            pickle.dump(cacheDatas,f)
        
        X = np.array(X,dtype=np.float32)
        Y = np.array(Y,dtype=np.float32)
        
        testNum = 5
        CUDA_X = torch.from_numpy(X[:-testNum]).cuda()
        CUDA_Y = torch.from_numpy(Y[:-testNum].reshape(Y.shape[0]-testNum,1,-1)).cuda()
        CUDA_XTEST = torch.from_numpy(X[-testNum:]).cuda()
        YTest = Y[-testNum:]
        Variable_X = Variable(CUDA_X)
        Variable_Y = Variable(CUDA_Y)
        Variable_XTEST = Variable(CUDA_XTEST)
        for epoch in range(totalEpoch):
            out = self.model(Variable_X)
            loss = criterion(out,Variable_Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1)%100 == 0:
                print(u'第%d次迭代循环后，剩余的拟合误差为：%.4f!'%((epoch+1),loss.data[0]))
                self.covKernelLoss()
        print('正在存储模型结构参数')
        torch.save(self.model.state_dict(),os.path.join(self.logDir,self.MainModelParas))
        self.model.eval()
        predict = self.model(Variable_XTEST).cpu().data.numpy()[:,0:1,0]
        predict = self.targetScaler.inverse_transform(predict)
        YTest = self.targetScaler.inverse_transform(YTest.reshape(-1,1))
        diff = np.mean(predict.ravel()) - np.mean(YTest.ravel())
        print(u'模型预测值为%.2f,实际值是%.2f'%(np.mean(predict.ravel()),np.mean(YTest.ravel())))
        print(u'模型当前的预测误差为%.2f'%diff)
        
        return diff

    # 控制周期训练
    def controlTrain(self,totalEpoch,corr):
        baseParasPath = self.MainModelParas
        # 生成四个基础模型参数文件
        model_1_ParaPath = os.path.join(self.logDir,'1_'+baseParasPath)
        if not os.path.exists(model_1_ParaPath):
            torch.save(self.model.state_dict(),model_1_ParaPath)
        model_2_ParaPath = os.path.join(self.logDir,'2_'+baseParasPath)
        if not os.path.exists(model_2_ParaPath):
            torch.save(self.model.state_dict(),model_2_ParaPath)
        model_3_ParaPath = os.path.join(self.logDir,'3_'+baseParasPath)
        if not os.path.exists(model_3_ParaPath):
            torch.save(self.model.state_dict(),model_3_ParaPath)
        model_4_ParaPath = os.path.join(self.logDir,'4_'+baseParasPath)
        if not os.path.exists(model_4_ParaPath):
            torch.save(self.model.state_dict(),model_4_ParaPath)
        # 加载数据
        X,Y,weatherData,ClockData = self.preprocess()
        
        cacheDatas = []
        for k,data in enumerate(zip(X,Y,weatherData,ClockData)):
            span = 600
            if k%span == 0:
                cacheData = []
                x,y,_,_ = data
                z = np.mean(weatherData[k-60:k],axis=0)
                c = np.mean(ClockData[k-60:k])
                x = x.ravel()
                cacheData.extend(x)
                cacheData.extend([y])
                cacheData.extend(z)
                cacheData.extend([c])
                cacheDatas.append(cacheData)
        cacheDatas = np.array(cacheDatas)
        if os.path.exists(os.path.join(self.logDir,self.CacheDataFile)):
            with open(os.path.join(self.logDir,self.CacheDataFile),'rb') as f:
                originalData = pickle.load(f)
            cacheDatas = np.vstack([originalData,cacheDatas])
        with open(os.path.join(self.logDir,self.CacheDataFile),'wb') as f:
            pickle.dump(cacheDatas,f)

        X = np.array(X,dtype=np.float32)
        Y = np.array(Y,dtype=np.float32)
        testNum = 5
        # 加载第一个子模型的参数
        self.model.load_state_dict(torch.load(model_1_ParaPath))
        # 对第一个模型进行训练
        WD = self.weight_decay
        LR = self.learningRate * corr[0]
        criterion = nn.L1Loss()
        optimizer = optim.Adam(self.model.parameters(),lr=LR,weight_decay=WD)
        CUDA_X = torch.from_numpy(X[:-testNum]).cuda()
        CUDA_Y = torch.from_numpy(Y[:-testNum].reshape(Y.shape[0]-testNum,1,-1)).cuda()
        CUDA_XTEST = torch.from_numpy(X[-testNum:]).cuda()
        YTest = Y[-testNum:]
        Variable_X = Variable(CUDA_X)
        Variable_Y = Variable(CUDA_Y)
        Variable_XTEST = Variable(CUDA_XTEST)
        for epoch in range(totalEpoch):
            out = self.model(Variable_X)
            loss = criterion(out,Variable_Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch + 1)%100 == 0:
                print(u'第%d次迭代循环后，第一子模型剩余的拟合误差为：%.4f!'%(epoch,loss.data[0]))
                for ttt in range(5):
                    self.covKernelLoss()
        for ttt in range(5):
            self.covKernelLoss()
        print(u'正在存储第一子模型结构参数')
        torch.save(self.model.state_dict(),model_1_ParaPath)
        self.model.eval()
        predict = self.model(Variable_XTEST).cpu().data.numpy()[:,0:1,0]
        predict = self.targetScaler.inverse_transform(predict)
        YTest_N = self.targetScaler.inverse_transform(YTest.reshape(-1,1))
        diff = np.mean(predict.ravel()) - np.mean(YTest_N.ravel())
        print(u'第一子模型预测值为%.2f,实际值是%.2f'%(np.mean(predict.ravel()),np.mean(YTest_N.ravel())))
        print(u'第一子模型当前的预测误差为%.2f'%diff)
        diff_1 = diff
        self.diff_1 = diff_1

        # 加载第二个子模型的参数
        self.model.load_state_dict(torch.load(model_2_ParaPath))
        # 对第二个模型进行训练
        WD = self.weight_decay
        LR = self.learningRate * corr[1]
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(),lr=LR,weight_decay=WD)       
        for epoch in range(totalEpoch):
            out = self.model(Variable_X)
            loss = criterion(out,Variable_Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch + 1)%100 == 0:
                print(u'第%d次迭代循环后，第二子模型剩余的拟合误差为：%.4f!'%(epoch,loss.data[0]))
                for ttt in range(5):
                    self.covKernelLoss()
        for ttt in range(5):
            self.covKernelLoss()
        print(u'正在存储第二子模型结构参数')
        torch.save(self.model.state_dict(),model_2_ParaPath)
        self.model.eval()
        predict = self.model(Variable_XTEST).cpu().data.numpy()[:,0:1,0]
        predict = self.targetScaler.inverse_transform(predict)
        YTest_N = self.targetScaler.inverse_transform(YTest.reshape(-1,1))
        diff = np.mean(predict.ravel()) - np.mean(YTest_N.ravel())
        print(u'第二子模型预测值为%.2f,实际值是%.2f'%(np.mean(predict.ravel()),np.mean(YTest_N.ravel())))
        print(u'第二子模型当前的预测误差为%.2f'%diff)
        diff_2 = diff
        self.diff_2 = diff_2

        # 加载第三个子模型的参数
        self.model.load_state_dict(torch.load(model_3_ParaPath))
        
        WD = self.weight_decay
        LR = self.learningRate * corr[2]
        criterion = nn.L1Loss()
        optimizer = optim.Adam(self.model.parameters(),lr=LR,weight_decay=WD)
        # 加载第三个子模型的数据
        # 必要的参数
        upLag = self.upLag
        inChannel = len(self.featureColumns) + 1
        weatherColumnsLen = len(self.WDColumnsNames.split(','))
        # 增加相关性样本
        if True:
            CacheDataPath = os.path.join(self.logDir,self.CacheDataFile)
            with open(CacheDataPath,'rb') as f:
                cacheData = pickle.load(f)
            cacheX = []
            cacheY = []
            cacheWeather = []
            cacheClock = []
            for line in cacheData:
                if np.isnan(sum(line)):
                    continue
                tempX = line[:upLag*(inChannel)]
                tempX = np.array(tempX)
                cacheX.append(tempX.reshape(-1,upLag))
                cacheY.append(line[upLag*(inChannel)])
                cacheWeather.append(line[upLag*(inChannel)+1:upLag*(inChannel)+weatherColumnsLen+1])
                cacheClock.append(line[upLag*(inChannel)+1+weatherColumnsLen])
            # 寻找风向上最相似的数据    
            targetWeather = np.mean(self.dataFrame[self.WDColumnsNames.split(',')].as_matrix()[-30:],axis=0)
            weatherScaler = preprocessing.StandardScaler()
            weatherScaler.fit(cacheWeather)
            cacheWeather = weatherScaler.transform(cacheWeather)
            targetWeather = weatherScaler.transform(targetWeather.reshape(1,-1))[0]
            wscores = [sum(targetWeather*i) for i in cacheWeather]
            addDataIndex_w = np.argsort(wscores)[-int(len(wscores)*0.1):]
            # 寻找时刻上最接近的数据
            targetClock = np.mean([i.hour for i in self.dataFrame.index[-10:]])
            cscores = np.argsort(np.array(cacheClock) - targetClock)
            addDataIndex_c = cscores[:int(len(cscores)*0.025)]
            addX_w = np.array(cacheX)[addDataIndex_w]
            addY_w = np.array(cacheY)[addDataIndex_w]
            addX_c = np.array(cacheX)[addDataIndex_c]
            addY_c = np.array(cacheY)[addDataIndex_c]
            X_3 = np.vstack([addX_w,X[-120*8:]])
            Y_3 = np.hstack([addY_w,Y[-120*8:]])

        X_3 = np.array(X_3,dtype=np.float32)
        Y_3 = np.array(Y_3,dtype=np.float32)

        # 对第三个模型进行训练
        CUDA_X_3 = torch.from_numpy(X_3[:-testNum]).cuda()
        CUDA_Y_3 = torch.from_numpy(Y_3[:-testNum].reshape(Y_3.shape[0]-testNum,1,-1)).cuda()
        Variable_X_3 = Variable(CUDA_X_3)
        Variable_Y_3 = Variable(CUDA_Y_3)
        for epoch in range(totalEpoch):
            out = self.model(Variable_X_3)
            loss = criterion(out,Variable_Y_3)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch + 1)%100 == 0:
                print(u'第%d次迭代循环后，第三子模型剩余的拟合误差为：%.4f!'%(epoch,loss.data[0]))
                for ttt in range(5):
                    self.covKernelLoss()
        for ttt in range(5):
            self.covKernelLoss()
        print(u'正在存储第三子模型结构参数')
        torch.save(self.model.state_dict(),model_3_ParaPath)
        self.model.eval()
        predict = self.model(Variable_XTEST).cpu().data.numpy()[:,0:1,0]
        predict = self.targetScaler.inverse_transform(predict)
        YTest_N = self.targetScaler.inverse_transform(YTest.reshape(-1,1))
        diff = np.mean(predict.ravel()) - np.mean(YTest_N.ravel())
        print(u'第三子模型预测值为%.2f,实际值是%.2f'%(np.mean(predict.ravel()),np.mean(YTest_N.ravel())))
        print(u'第三子模型当前的预测误差为%.2f'%diff)
        diff_3 = diff
        self.diff_3 = diff_3
        
        # 加载第四个子模型的参数
        self.model.load_state_dict(torch.load(model_4_ParaPath))

        # 加载第四个子模型的数据
        WD = self.weight_decay
        LR = self.learningRate * corr[3]
        criterion = nn.MSELoss()
        #optimizer = optim.Adam(self.model.parameters(),lr=LR,weight_decay=WD)
        optimizer = optim.Adam(self.model.parameters(),lr=LR)
        '''
        # 增加相关性样本
        if True:
            with open('CacheData.z') as f:
                cacheData = pickle.load(f)
            cacheX = []
            cacheY = []
            cacheWeather = []
            cacheClock = []
            for line in cacheData:
                if np.isnan(sum(line)):
                    continue
                tempX = line[:upLag*(inChannel+1)]
                tempX = np.array(tempX)
                cacheX.append(tempX.reshape(-1,upLag))
                cacheY.append(line[upLag*(inChannel+1)])
                cacheWeather.append(line[upLag*(inChannel+1)+1:upLag*(inChannel+1)+weatherColumnsLen+1])
                cacheClock.append(line[upLag*(inChannel+1)+1+weatherColumnsLen])
            # 寻找风向上最相似的数据    
            targetWeather = np.mean(self.dataFrame[self.WDColumnsNames.split(',')].as_matrix()[-30:],axis=0)
            weatherScaler = preprocessing.StandardScaler()
            weatherScaler.fit(cacheWeather)
            cacheWeather = weatherScaler.transform(cacheWeather)
            targetWeather = weatherScaler.transform(targetWeather.reshape(1,-1))[0]
            wscores = [sum(targetWeather*i) for i in cacheWeather]
            addDataIndex_w = np.argsort(wscores)[-int(len(wscores)*0.05):]
            # 寻找时刻上最接近的数据
            targetClock = np.mean([i.hour for i in self.dataFrame.index[-10:]])
            cscores = np.argsort(np.array(cacheClock) - targetClock)
            addDataIndex_c = cscores[:int(len(cscores)*0.05)]
            addX_w = np.array(cacheX)[addDataIndex_w]
            addY_w = np.array(cacheY)[addDataIndex_w]
            addX_c = np.array(cacheX)[addDataIndex_c]
            addY_c = np.array(cacheY)[addDataIndex_c]
            X_4 = np.vstack([addX_w,addX_c,X[-120*12:]])
            Y_4 = np.hstack([addY_w,addY_c,Y[-120*12:]])
        '''
        X_4 = X
        Y_4 = Y
        X_4 = np.array(X_4,dtype=np.float32)
        Y_4 = np.array(Y_4,dtype=np.float32)
        
        # 对第四个模型进行训练
        CUDA_X_4 = torch.from_numpy(X_4[:-testNum]).cuda()
        CUDA_Y_4 = torch.from_numpy(Y_4[:-testNum].reshape(Y_4.shape[0]-testNum,1,-1)).cuda()
        Variable_X_4 = Variable(CUDA_X_4)
        Variable_Y_4 = Variable(CUDA_Y_4)
        for epoch in range(totalEpoch):
            out = self.model(Variable_X_4)
            loss = criterion(out,Variable_Y_4)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch + 1)%100 == 0:
                print(u'第%d次迭代循环后，第四子模型剩余的拟合误差为：%.4f!'%(epoch,loss.data[0]))
                for ttt in range(5):
                    self.covKernelLoss()
        for ttt in range(5):
            self.covKernelLoss()
        print(u'正在存储第四子模型结构参数')
        torch.save(self.model.state_dict(),model_4_ParaPath)
        self.model.eval()
        predict = self.model(Variable_XTEST).cpu().data.numpy()[:,0:1,0]
        predict = self.targetScaler.inverse_transform(predict)
        YTest_N = self.targetScaler.inverse_transform(YTest.reshape(-1,1))
        diff = np.mean(predict.ravel()) - np.mean(YTest_N.ravel())
        print(u'第四子模型预测值为%.2f,实际值是%.2f'%(np.mean(predict.ravel()),np.mean(YTest_N.ravel())))
        print(u'第四子模型当前的预测误差为%.2f'%diff)
        diff_4 = diff
        self.diff_4 = diff_4
        
        scoreFilePath = os.path.join(self.logDir,self.scoresFile)
        if os.path.exists(scoreFilePath):
            with open(scoreFilePath,'r') as f:
                ftext = f.read()
                wcorr = [float(i) for i in ftext.split(',')]
                wcorr = np.array(wcorr)/sum(wcorr)
        else:
            wcorr = np.array([0.25,0.25,0.25,0.25])
        
        print(u'分配权重'),
        print(wcorr/sum(wcorr))
        print(u'误差列表'),
        print([diff_1,diff_2,diff_3,diff_4])
        print(u'总模型预测误差为：%.2f'%sum(wcorr/sum(wcorr) * np.array([diff_1,diff_2,diff_3,diff_4])))
        return sum(wcorr/sum(wcorr) * np.array([diff_1,diff_2,diff_3,diff_4]))

    def predictModel(self,predictX,pLength,testLabel=False,testLength=0,predictIndex=0):
        if predictIndex == 0:
            if os.path.exists('P_1.txt'):
                os.remove('P_1.txt')
            if os.path.exists('P_2.txt'):
                os.remove('P_2.txt')
            if os.path.exists('P_3.txt'):
                os.remove('P_3.txt')
            if os.path.exists('P_4.txt'):
                os.remove('P_4.txt')
            
        
        # 产生新的日期索引
        currentIndex = list(self.NoNull_dataFrame.index)
        lastDate = currentIndex[-1]
        if testLabel:
            stopDate = lastDate + datetime.timedelta(0,60*testLength)
        else:
            stopDate = lastDate + datetime.timedelta(0,60*pLength)
        startDate = lastDate + datetime.timedelta(0,60)
        newIndex = pd.date_range(start=startDate,end=stopDate,freq='1min')
        # 合并虚拟数据和实际数据
        newDataFrame = pd.DataFrame(predictX,index=newIndex,columns=self.featureColumns)
        validDataFrame = self.NoNull_dataFrame[self.featureColumns]
        newDataFrame = pd.concat([validDataFrame,newDataFrame])
        newDataFrame = newDataFrame.fillna(method='ffill').fillna(method='bfill')

        # 对数据结构进行转换
        newpredictX = newDataFrame[self.featureColumns].as_matrix()
        tempY = np.random.randn(newpredictX.shape[0],1)
        
        newpredictX,tempY = self.transDataFrame(newDataFrame,newpredictX,tempY,True)
        # 预测
        newpredictX = np.array(newpredictX,dtype=np.float32)
        Variable_PredictX = Variable(torch.from_numpy(newpredictX).cuda())
        baseParasPath = self.MainModelParas
        model_1_path = os.path.join(self.logDir,'1_'+baseParasPath)
        model_2_path = os.path.join(self.logDir,'2_'+baseParasPath)
        model_3_path = os.path.join(self.logDir,'3_'+baseParasPath)
        model_4_path = os.path.join(self.logDir,'4_'+baseParasPath)
        # 第一子模型进行预测
        self.model.load_state_dict(torch.load(model_1_path))
        predictY = self.model(Variable_PredictX)
        predictY = predictY.cpu().data.numpy()
        predictY = predictY.reshape(predictY.shape[0],predictY.shape[2])
        predictY = self.targetScaler.inverse_transform(predictY)
        predictY_1 = predictY[-predictX.shape[0]:]
        predictY_1 = predictY_1.ravel()
        with open('P_1.txt','a') as f:
            s = ','.join([str(i) for i in predictY_1-self.diff_1])
            f.write(s + '\n')

        # 第二子模型进行预测
        self.model.load_state_dict(torch.load(model_2_path))
        predictY = self.model(Variable_PredictX)
        predictY = predictY.cpu().data.numpy()
        predictY = predictY.reshape(predictY.shape[0],predictY.shape[2])
        predictY = self.targetScaler.inverse_transform(predictY)
        predictY_2 = predictY[-predictX.shape[0]:]
        predictY_2 = predictY_2.ravel()
        with open('P_2.txt','a') as f:
            s = ','.join([str(i) for i in predictY_2-self.diff_2])
            f.write(s + '\n')
        # 第三子模型进行预测
        self.model.load_state_dict(torch.load(model_3_path))
        predictY = self.model(Variable_PredictX)
        predictY = predictY.cpu().data.numpy()
        predictY = predictY.reshape(predictY.shape[0],predictY.shape[2])
        predictY = self.targetScaler.inverse_transform(predictY)
        predictY_3 = predictY[-predictX.shape[0]:]
        predictY_3 = predictY_3.ravel()
        with open('P_3.txt','a') as f:
            s = ','.join([str(i) for i in predictY_3-self.diff_3])
            f.write(s + '\n')
            
        # 第四子模型进行预测
        self.model.load_state_dict(torch.load(model_4_path))
        predictY = self.model(Variable_PredictX)
        predictY = predictY.cpu().data.numpy()
        predictY = predictY.reshape(predictY.shape[0],predictY.shape[2])
        predictY = self.targetScaler.inverse_transform(predictY)
        predictY_4 = predictY[-predictX.shape[0]:]
        predictY_4 = predictY_4.ravel()
        with open('P_4.txt','a') as f:
            s = ','.join([str(i) for i in predictY_4-self.diff_4])
            f.write(s + '\n')

        scoreFilePath = os.path.join(self.logDir,self.scoresFile)
        if os.path.exists(scoreFilePath):
            with open(scoreFilePath,'r') as f:
                ftext = f.read()
                wcorr = [float(i) for i in ftext.split(',')]
                wcorr = np.array(wcorr)/sum(wcorr)
        else:
            wcorr = np.array([0.25,0.25,0.25,0.25])
        return predictY_1 * wcorr[0] + predictY_2 * wcorr[1] + predictY_3 * wcorr[2] + predictY_4 * wcorr[3]
        



class CNNMode_Kernal_2(nn.Module):
    def __init__(self,inChannel,outChannel,step,hidden,upLag,outPut):
        super(CNNMode_Kernal_2,self).__init__()
        self.inChannel = inChannel
        i = 0
        if i < inChannel:
            self.layer_10 = nn.Conv1d(1,1,step,step,padding=0)
            i = i+1
        if i < inChannel:
            self.layer_11 = nn.Conv1d(1,1,step,step,padding=0)
            i = i+1
        if i < inChannel:
            self.layer_12 = nn.Conv1d(1,1,step,step,padding=0)
            i = i+1
        if i < inChannel:
            self.layer_13 = nn.Conv1d(1,1,step,step,padding=0)
            i = i+1
        if i < inChannel:
            self.layer_14 = nn.Conv1d(1,1,step,step,padding=0)
            i = i+1
        if i < inChannel:
            self.layer_15 = nn.Conv1d(1,1,step,step,padding=0)
            i = i+1
        if i < inChannel:
            self.layer_16 = nn.Conv1d(1,1,step,step,padding=0)
            i = i+1
        if i < inChannel:
            self.layer_17 = nn.Conv1d(1,1,step,step,padding=0)
            i = i+1
        if i < inChannel:
            self.layer_18 = nn.Conv1d(1,1,step,step,padding=0)
            i = i+1
        if i < inChannel:
            self.layer_19 = nn.Conv1d(1,1,step,step,padding=0)
            i = i+1
        if i < inChannel:
            self.layer_110 = nn.Conv1d(1,1,step,step,padding=0)
            i = i+1
        if i < inChannel:
            self.layer_111 = nn.Conv1d(1,1,step,step,padding=0)
            i = i+1
            
        self.layer_2 = nn.Sequential(
                                     nn.Linear(int(inChannel*(upLag)/float(step)),hidden),nn.ReLU(True))
        self.layer_3 = nn.Linear(hidden,outPut)
        
    def forward(self,x):
        subX = []
        
        i = 0
        if i < self.inChannel:
            partX = x[:,i:i+1,:]
            partXOut = self.layer_10(partX)
            s,c,f = partXOut.size()
            partXOut = partXOut.view(s,c*f)
            subX.append(partXOut)
            i = i + 1
        if i < self.inChannel:
            partX = x[:,i:i+1,:]
            partXOut = self.layer_11(partX)
            s,c,f = partXOut.size()
            partXOut = partXOut.view(s,c*f)
            subX.append(partXOut)
            i = i + 1
        if i < self.inChannel:
            partX = x[:,i:i+1,:]
            partXOut = self.layer_12(partX)
            s,c,f = partXOut.size()
            partXOut = partXOut.view(s,c*f)
            subX.append(partXOut)
            i = i + 1
        if i < self.inChannel:
            partX = x[:,i:i+1,:]
            partXOut = self.layer_13(partX)
            s,c,f = partXOut.size()
            partXOut = partXOut.view(s,c*f)
            subX.append(partXOut)
            i = i + 1
        if i < self.inChannel:
            partX = x[:,i:i+1,:]
            partXOut = self.layer_14(partX)
            s,c,f = partXOut.size()
            partXOut = partXOut.view(s,c*f)
            subX.append(partXOut)
            i = i + 1
        if i < self.inChannel:
            partX = x[:,i:i+1,:]
            partXOut = self.layer_15(partX)
            s,c,f = partXOut.size()
            partXOut = partXOut.view(s,c*f)
            subX.append(partXOut)
            i = i + 1
        if i < self.inChannel:
            partX = x[:,i:i+1,:]
            partXOut = self.layer_16(partX)
            s,c,f = partXOut.size()
            partXOut = partXOut.view(s,c*f)
            subX.append(partXOut)
            i = i + 1
        if i < self.inChannel:
            partX = x[:,i:i+1,:]
            partXOut = self.layer_17(partX)
            s,c,f = partXOut.size()
            partXOut = partXOut.view(s,c*f)
            subX.append(partXOut)
            i = i + 1
        if i < self.inChannel:
            partX = x[:,i:i+1,:]
            partXOut = self.layer_18(partX)
            s,c,f = partXOut.size()
            partXOut = partXOut.view(s,c*f)
            subX.append(partXOut)
            i = i + 1
        if i < self.inChannel:
            partX = x[:,i:i+1,:]
            partXOut = self.layer_19(partX)
            s,c,f = partXOut.size()
            partXOut = partXOut.view(s,c*f)
            subX.append(partXOut)
            i = i + 1
        if i < self.inChannel:
            partX = x[:,i:i+1,:]
            partXOut = self.layer_110(partX)
            s,c,f = partXOut.size()
            partXOut = partXOut.view(s,c*f)
            subX.append(partXOut)
            i = i + 1
        if i < self.inChannel:
            partX = x[:,i:i+1,:]
            partXOut = self.layer_111(partX)
            s,c,f = partXOut.size()
            partXOut = partXOut.view(s,c*f)
            subX.append(partXOut)
            i = i + 1
        subX = torch.cat(tuple(subX),1)
        X = self.layer_2(subX)
        X = self.layer_3(X)
        X = X.view(s,1,-1)
        return X        