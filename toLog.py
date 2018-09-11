# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 16:57:55 2018

@author: 98089
"""
import os

def toLog(level,part,content):
    if not os.path.exists('logs'):
        os.mkdir('logs')
    with open('logs/log.csv',"a") as f:
        f.write('(%s)[%s]:%s\n'%(part,level,content))
        print('(%s)[%s]:%s\n'%(part,level,content))
        
