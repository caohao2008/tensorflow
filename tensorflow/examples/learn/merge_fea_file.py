#!/bin/python

import os
import sys

feature_mapping=""
i=0
for line in open(sys.argv[1]):
    feature_mapping=feature_mapping+",\""+line.strip()+"\""
    i=i+1
    if i%5 ==0:
        feature_mapping+="\n"
print(feature_mapping)
