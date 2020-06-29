
import sys
sys.path.append("../")

import json
import os
import argparse
import torch
import shutil
import numpy as np

num_loop = 1

#for i in range(num_loop):
#    op = 'CUDA_VISIBLE_DEVICES=2 python ar_test.py -i 0 -em test -analyze -ns -write_time -beam_size 1'
#    os.system(op)

'''
for bs in [1, 5, 6]:
    op = 'CUDA_VISIBLE_DEVICES=0 python ar_test.py -i 1 -em test -analyze -write_time -beam_size %d'%bs
    os.system(op)


for iteration in range(1, 8):
    for i in range(num_loop):
        op = 'CUDA_VISIBLE_DEVICES=0 python test_nar.py --index 1 -beam_alpha 1.35 -em test -nd -paradigm mp -print_latency -write_time -lbs 6 -s 100 ' + ' -i %d'%iteration
        os.system(op)

for iteration in range(1, 8):
    for i in range(num_loop):
        op = 'CUDA_VISIBLE_DEVICES=0 python test_nar.py --index -1 -beam_alpha 1.35 -em test -nd -paradigm mp -print_latency -write_time -lbs 6 ' + ' -i %d'%iteration
        os.system(op)

#myset = [[5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [4, 4], [3, 3], [2, 2], [1, 1]]
myset = [[3, 4], [3, 5]]
for item in myset:
    i, lbs = item
    op = 'CUDA_VISIBLE_DEVICES=0 python test_nar.py --index 1 -beam_alpha 1.35 -em test -nd -paradigm mp -s 100 -print_latency -write_time' + ' -i %d'%i + ' -lbs %d'%lbs
    os.system(op)


for q in range(4, 0, -1):
    for iteration in range(2):
        for i in range(num_loop):
            op = 'CUDA_VISIBLE_DEVICES=0 python test_nar.py --index 1 -beam_alpha 1.35 -em test -nd -paradigm ef -s 100 -print_latency -write_time -lbs 6 ' + ' -i %d'%iteration + ' -q %d'%q
            os.system(op)
'''



for bs in [1, 5, 6]:
    op = 'CUDA_VISIBLE_DEVICES=0 python ar_test.py -i 0 -em test -analyze -write_time -beam_size %d'%bs
    os.system(op)


for iteration in range(1, 8):
    for i in range(num_loop):
        op = 'CUDA_VISIBLE_DEVICES=0 python test_nar.py --index 0 -em test -nd -paradigm mp -print_latency -write_time -lbs 6 -s 100 ' + ' -i %d'%iteration
        os.system(op)

for iteration in range(1, 8):
    for i in range(num_loop):
        op = 'CUDA_VISIBLE_DEVICES=0 python test_nar.py --index -2 -em test -nd -paradigm mp -print_latency -write_time -lbs 6 ' + ' -i %d'%iteration
        os.system(op)

#myset = [[5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [4, 4], [3, 3], [2, 2], [1, 1]]
myset = [[1,1], [1, 2],[1, 3], [1, 4],[1, 5]]
for item in myset:
    i, lbs = item
    op = 'CUDA_VISIBLE_DEVICES=0 python test_nar.py --index 0 -em test -nd -paradigm mp -s 100 -print_latency -write_time' + ' -i %d'%i + ' -lbs %d'%lbs
    os.system(op)


for q in range(4, 0, -1):
    for iteration in range(2):
        for i in range(num_loop):
            op = 'CUDA_VISIBLE_DEVICES=0 python test_nar.py --index 0 -em test -nd -paradigm ef -s 100 -print_latency -write_time -lbs 6 ' + ' -i %d'%iteration + ' -q %d'%q
            os.system(op)






'''


for i in range(num_loop):
    op = 'CUDA_VISIBLE_DEVICES=0 python ar_test.py -i 0 -em test -analyze -ns -write_time -beam_size 1'
    os.system(op)

for i in range(num_loop):
    op = 'CUDA_VISIBLE_DEVICES=0 python ar_test.py -i 0 -em test -analyze -ns -write_time'
    os.system(op)

for iteration in range(1, 8):
    for i in range(num_loop):
        op = 'CUDA_VISIBLE_DEVICES=0 python test_nar.py -ns --index 0 -em test -nd -paradigm mp -s 100 -print_latency -write_time' + ' -i %d'%iteration
        os.system(op)

for iteration in range(1, 8):
    for i in range(num_loop):
        op = 'CUDA_VISIBLE_DEVICES=0 python test_nar.py -ns --index -2 -em test -nd -paradigm mp -s 100 -print_latency -write_time' + ' -i %d'%iteration
        os.system(op)
'''
'''
myset = [[5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [4, 4], [3, 3], [2, 2], [1, 1]]
for item in myset:
    i, lbs = item
    op = 'CUDA_VISIBLE_DEVICES=0 python test_nar.py --index 0 -em test -nd -paradigm mp -s 100 -print_latency -write_time' + ' -i %d'%i + ' -lbs %d'%lbs
    os.system(op)

for q in range(4, 0, -1):
    for iteration in range(2):
        for i in range(num_loop):
            op = 'CUDA_VISIBLE_DEVICES=0 python test_nar.py --index 0 -em test -nd -paradigm ef -s 100 -print_latency -write_time' + ' -i %d'%iteration + ' -q %d'%q
            os.system(op)

for i in range(num_loop):
    op = 'CUDA_VISIBLE_DEVICES=3 python ar_test.py -i 1 -em test -analyze -ns -write_time -beam_size 1'
    os.system(op)

for i in range(num_loop):
    op = 'CUDA_VISIBLE_DEVICES=3 python ar_test.py -i 1 -em test -analyze -ns -write_time'
    os.system(op)


for iteration in range(1, 7):
    for i in range(num_loop):
        op = 'CUDA_VISIBLE_DEVICES=3 python test_nar.py --index 0 -em test -nd -paradigm mp -s 100 -print_latency -write_time -ns' + ' -i %d'%iteration
        os.system(op)
for iteration in range(5, 7):
    for i in range(num_loop):
        op = 'CUDA_VISIBLE_DEVICES=3 python test_nar.py --index 0 -em test -nd -paradigm mp -print_latency -write_time -ns' + ' -i %d'%iteration
        os.system(op)



for iteration in range(5, 7):
    for i in range(num_loop):
        op = 'CUDA_VISIBLE_DEVICES=3 python test_nar.py --index 1 -beam_alpha 1.15 -em test -nd -paradigm mp -print_latency -write_time -ns' + ' -i %d'%iteration
        os.system(op)

for i in range(num_loop):
    op = 'CUDA_VISIBLE_DEVICES=3 python test_nar.py --index 1 -beam_alpha 1.15 -em test -nd -paradigm ef -print_latency -write_time -ns'
    os.system(op)

for i in range(num_loop):
    op = 'CUDA_VISIBLE_DEVICES=3 python test_nar.py --index 1 -beam_alpha 1.15 -em test -nd -paradigm ef -s 100 -print_latency -write_time -ns'
    os.system(op)

for i in range(num_loop):
    op = 'CUDA_VISIBLE_DEVICES=3 python test_nar.py --index 1 -beam_alpha 1.15 -em test -nd -paradigm ef -s 100 -print_latency -write_time -ns -q 2'
    os.system(op)

for iteration in range(1, 7):
    for i in range(num_loop):
        op = 'CUDA_VISIBLE_DEVICES=3 python test_nar.py --index 0 -em test -nd -paradigm mp -s 100 -print_latency -write_time -i 5 ' + ' -lbs %d'%iteration
        if i > 0:
            op += " -ns "
        os.system(op)
for iteration in range(3, 7):
    for i in range(num_loop):
        op = 'CUDA_VISIBLE_DEVICES=3 python test_nar.py --index 1 -em test -nd -paradigm mp -beam_alpha 1.15 -s 100 -print_latency -write_time -i 5 ' + ' -lbs %d'%iteration
        if i > 0:
            op += " -ns "
        os.system(op)
for iteration in range(1, 7):
    for i in range(num_loop):
        op = 'CUDA_VISIBLE_DEVICES=3 python test_nar.py --index -1 -em test -nd -paradigm mp -beam_alpha 1.15 -print_latency -write_time -i 5 ' + ' -lbs %d'%iteration
        if i > 0:
            op += " -ns "
        os.system(op)

for iteration in range(1, 7):
    for i in range(num_loop):
        op = 'CUDA_VISIBLE_DEVICES=3 python test_nar.py --index 0 -s 100 -em test -nd -paradigm mp -print_latency -write_time -lbs 5 ' + ' -i %d'%iteration
        #if i > 0:
        #    op += " -ns "
        os.system(op)
'''