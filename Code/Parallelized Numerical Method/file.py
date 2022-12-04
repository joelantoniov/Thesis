#! /usr/bin/env python
# -*- coding: utf-8 -*-
import random

for i in range(1, 1001):
    print "6", "1", "0.00001", random.randint(-100, -70), random.randint(70, 100), "0.01", random.sample([0.001, 0.01, 0.1, 0.0001], 1)[0], random.sample([0.001, 0.01], 1)[0]
