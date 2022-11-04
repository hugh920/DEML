#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--cuda', action='store_true',default=True, help='enables cuda')
# parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

parser.add_argument('--job_id', type=str, default='14567', help='file job id')

parser.add_argument('--heads', type=int, default=4, help='Heads for region Atn')

parser.add_argument('--cosinelr_scheduler', action='store_true',default=False, help='Run with lr scheduler')
parser.add_argument('--summary', type=str, default='Summary', help='Summary of Expt')
parser.add_argument('--src', type=str,default="datasets")

parser.add_argument('--channel_dim', type=int, default=512,help='conv channel dim')


opt = parser.parse_args()