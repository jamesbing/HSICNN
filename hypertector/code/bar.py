#!/usr/bin/env python
# coding=utf-8
import sys,time
class Bar:
    def __init__(self,count = 0, total = 0, width = 100):
        self.count = count
        self.total = total
        self.width = width

    def move(self):
        self.count += 1

    def log(self, s):
        sys.stdout.write(' ' * (self.width + 9) + '\r')
        sys.stdout.flush()
        print s
        progress = self.width * self.count / self.total
        sys.stdout.write('\033[5;31;0m')
        sys.stdout.write('{0:3}/{1:3}: '.format(self.count, self.total))
#        sys.stdout.write('#' * progress + '-' * (self.width - progress) + '\r')
        sys.stdout.write(chr(219) * progress + '-' * (self.width - progress) + '\r')
        if progress == self.width:
            sys.stdout.write('\n')
        sys.stdout.flush()
        sys.stdout.write('\033[0m')


