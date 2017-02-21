#!/usr/bin/env python
# coding=utf-8
from PIL import Image
import re

x = 503
y = 122

im = Image.new("RGB",(x,y))

for i in range(0,x):
    for j in range(0,y):
        im.putpixel((i,j),(i, j, i + j))

im.save("test.jpeg","JPEG")
