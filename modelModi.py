import os
import sys
import time
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from model import MyNet, ResNet18
from dataset import get_dataloader
from utils import set_seed, write_config_log, write_result_log

import config as cfg




logfile_dir='./experiment/resnet18_2023_04_14_16_03_33_sgd_pre_da/log/result_log.txt'

def plot_learning_curve(logfile_dir, result_lists):
    f= open(logfile_dir, 'r')
    train_acc=[]
    train_loss=[]
    val_acc=[]
    val_loss=[]
    epo=0
    for line in f.readlines():
        epo+=1
        train_acc.append(float(line[line.find('Train Acc')+11:line.find('Train Acc')+18]))
        train_loss.append(float(line[line.find('Train Loss')+12:line.find('Train Loss')+19]))
        val_acc.append(float(line[line.find('Val Acc')+9:line.find('Val Acc')+17]))
        val_loss.append(float(line[line.find('Val Loss')+10:line.find('Val Loss')+17]))
    epoch=np.arange(1,epo+1)


    fig = plt.figure()
    plt.subplot(221)
    f1 = np.polyfit(epoch, train_acc, 5)
    p1 = np.poly1d(f1)
    yvals1=p1(epoch)
    plt.plot(epoch, train_acc, 'ro', label='train acc')
    plt.plot(epoch, yvals1, 'r',label='polyfit train acc')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy')
    plt.legend()
    # plt.show()
    # f.close()
    plt.subplot(222)
    f1 = np.polyfit(epoch, train_loss, 5)
    p1 = np.poly1d(f1)
    yvals1=p1(epoch)
    plt.plot(epoch, train_loss, 'ro', label='train loss')
    plt.plot(epoch, yvals1, 'r',label='polyfit train loss')
    plt.xlabel('epoch')
    plt.ylabel('Error')
    plt.title('Train loss')
    plt.legend()
    # plt.show()
    # f.close()

    plt.subplot(223)
    f1 = np.polyfit(epoch, val_acc, 5)
    p1 = np.poly1d(f1)
    yvals1=p1(epoch)
    plt.plot(epoch, val_acc, 'ro', label='Val Accuracy')
    plt.plot(epoch, yvals1, 'r',label='polyfit Val Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.title('Val Accuracy')
    plt.legend()
    # plt.show()
    # f.close()

    plt.subplot(224)
    f1 = np.polyfit(epoch, val_loss, 5)
    p1 = np.poly1d(f1)
    yvals1=p1(epoch)
    plt.plot(epoch, val_loss, 'ro', label='Val Loss')
    plt.plot(epoch, yvals1, 'r',label='polyfit Val Loss')
    plt.xlabel('epoch')
    plt.ylabel('Error')
    plt.title('Val Loss')
    plt.legend()
    # plt.show()
    # f.close()

    plt.show()




current_result_lists = {'train_acc','train_loss','val_acc','val_loss'}

plot_learning_curve(logfile_dir, current_result_lists)
