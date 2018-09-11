# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 18:02:28 2018

@author: 98089
"""

import datetime,time,re

def tools_DateTimeTrans(ins_datetime):
    if isinstance(ins_datetime,datetime.datetime):
        return ins_datetime
    elif isinstance(ins_datetime,str):
        timePattern_1 = '(\d{2,4})-(\d{1,2})-(\d{1,2}) (\d{1,2}):(\d{1,2}):(\d{1,2})'
        if re.match(timePattern_1,ins_datetime):
            year,month,day,hour,minute,second = re.findall(timePattern_1,ins_datetime)[0]
            return datetime.datetime(int(year),int(month),int(day),int(hour),
                                     int(minute),int(second))
        else:
            raise ValueError('时间字符串格式不匹配')
    else:
        raise ValueError('时间对象格式不匹配')
        
        