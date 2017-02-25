#!/usr/bin/env python
# coding=utf-8
#用于画高光谱图像分类结果的RGB图
from PIL import Image,ImageDraw, ImageFont
import re

prompt = ">"

def getColors():
    #生成颜色列表
    colorsFile = open('resources/colors.txt')
    colorsDict = {}
    line = colorsFile.readline()
    putMark = 0
    while line:
        rgb = line.split(" ")
        line = colorsFile.readline()
        colorsDict[putMark] = rgb
        putMark = putMark + 1
    return colorsDict

def draw_box(img, x, y, rgb):
    for i in range(80):
        for j in range(20):
            img.putpixel((x + i , y + j),(int(rgb[0]), int(rgb[1]), int(rgb[2])))

def drawRGBResultCutline(filePath, classesNumber):
    # draw the cutline image fot the result image
    row = 200
    line = 400
    row_dist = 50
    line_dist = 10

    location_line = line / classesNumber

    colorsDict = getColors()
    cutline = Image.new("RGB", (row, line), (255,255,255))
    
    #font = ImageFont.truetype('simsun.ttc',24)
    draw = ImageDraw.Draw(cutline)

    classCount = 0
    y = 5
    for location in range(int(classesNumber)):
        rgb = colorsDict[classCount]
        x = 0
        draw.text((x + 10, y), unicode(str(classCount + 1)), outline = 'black')
        draw_box(cutline, x + 80, y, rgb)
        y = y + int(location_line)
        classCount = classCount + 1
    cutline.save(filePath + "_cutline.bmp", "BMP")

def drawRGBResult(fileName, labels, positions, raws, lines):
    im = Image.new("RGB", (raws, lines))
    positionMark = 0
    colorsDict = getColors()
    #rgb = []

    for result in labels:
    #    rgb = []
    #    if colorsDict.get(result, -1024) == -1024:
    #        line = colorsFile.readline()
    #        rgb = line.split(" ")
    #        colorsDict[result] = rgb
    #    else:
    #        rgb = colorsDict[result]
        position_temp = positions[positionMark]
        # for testing purpose: print position_temp
        position_pair = position_temp.split("|")
        x = int(position_pair[0])
        y = int(position_pair[1])
        #z = int(position_pair[2])
        #x = position_temp['row']
        #y = position_temp['line']
        rgb = colorsDict[int(result)]
        im.putpixel((x,y),(int(rgb[0]), int(rgb[1]), int(rgb[2])))
        #log_msg = "result: " + str(result) + "position:" + str(x) + "-" + str(y) + ", rgb:" + str(rgb)
        #color_drawing_log.write(log_msg + '\r')
        positionMark = positionMark + 1

    #color_drawing_log.close()
    im.save(fileName + ".bmp", "BMP")

   
def analyse():
    print "==============================Analyse============================="
    print "Chose analyse type:"
    print "#1 Draw RGB classifiation pictures; #2 TODO; #3 TODO; #4 TODODO"
    analyse_type = int(raw_input(prompt))
    if analyse_type == 1:
        print "Draw RGB Classification Picture(s) for"
        print "#1 only one result dataset or"
        print "#2 a batch of datasets"
        picture_type = raw_input(prompt)
        if picture_type == '1':

        elif picture_type == '2':

    elif analyse_type == 2:
        print "todo"
    elif analyse_type == 3:
        print "todo"
    elif analyse_type == 4:
        print "todo"


