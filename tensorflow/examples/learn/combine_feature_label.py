#!/bin/python

import os
import sys

file2=open(sys.argv[2])
for line in open(sys.argv[1]):
    line2 = file2.readline().strip()
    print line2+","+line.strip()
