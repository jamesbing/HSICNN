#!/usr/bin/env python
# coding=utf-8
#用于画高光谱图像分类结果的RGB图
from PIL import Image
import re

def drawRGBResult(fileName, labels, positions, raws, lines):
    im = Image.new("RGB", (raws, lines))
    
    #生成颜色列表
    colorsFile = open('colors.txt')
    colorsDict = {}
    
    color_drawing_log = open("/home/para/test.txt", 'w')

    positionMark = 0

    for result in labels:
        rgb = []
        if colorsDict.get(result, -1024) == -1024:
            line = colorsFile.readline()
            rgb = line.split(" ")
            colorsDict[result] = rgb
        else:
            rgb = colorsDict[result]
        position_temp = positions[positionMark]
        # for testing purpose: print position_temp
        position_pair = position_temp.split("|")
        x = int(position_pair[0])
        y = int(position_pair[1])
        #x = position_temp['row']
        #y = position_temp['line']
        im.putpixel((x,y),(int(rgb[0]), int(rgb[1]), int(rgb[2])))
        log_msg = "result: " + str(result) + "position:" + str(x) + "-" + str(y) + ", rgb:" + str(rgb)
        color_drawing_log.write(log_msg + '\r')
        positionMark = positionMark + 1
    color_drawing_log.close()
    im.save(fileName, "JPEG")

    
