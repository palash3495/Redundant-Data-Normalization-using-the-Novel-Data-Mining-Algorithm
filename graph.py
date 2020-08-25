from _lsprof import profiler_entry

import numpy as np


from normality.Result import *
import matplotlib.pyplot as plt


def accuracy():
    objects = ('FBR', 'WBorda', 'UDD', 'PEBL', 'LCS', 'GFS')
    y_pos = np.arange(len(objects))
    # Classifiers = ['KDD', 'NSL']
    bar_width = 0.70
    performance = [0.73, 0.80, 0.919, 0.851, LCS_Accuracy, GFS_Accuracy]
    # performance1 = [5, 15, 12, 45, 54]
    # plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.bar(y_pos, performance, bar_width, color='#375E97', edgecolor='black')
    # plt.bar(y_pos + bar_width, performance1, bar_width, color='#FB6542', edgecolor='black')

    plt.xticks(y_pos, objects, fontsize=10, rotation=0)
    plt.xlabel('Methods', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.title('Accuracy Graph')
    # plt.legend(Classifiers, loc=2)
    plt.show()

def precision():
    objects = ('FBR', 'WBorda', 'UDD', 'PEBL', 'LCS', 'GFS')
    y_pos = np.arange(len(objects))
    # Classifiers = ['KDD', 'NSL']
    bar_width = 0.70
    performance = [0.69, 0.78, 0.924, 0.902, LCS_Precision, GFS_Precision]
    # performance1 = [5, 15, 12, 45, 54]
    # plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.bar(y_pos, performance, bar_width, color='#375E97', edgecolor='black')
    # plt.bar(y_pos + bar_width, performance1, bar_width, color='#FB6542', edgecolor='black')

    plt.xticks(y_pos, objects, fontsize=10, rotation=0)
    plt.xlabel('Methods', fontsize=10)
    plt.ylabel('Precision', fontsize=10)
    plt.title('Precision Graph')
    # plt.legend(Classifiers, loc=2)
    plt.show()

def recall():
    objects = ('FBR', 'WBorda', 'UDD', 'PEBL', 'LCS', 'GFS')
    y_pos = np.arange(len(objects))
    # Classifiers = ['KDD', 'NSL']
    bar_width = 0.70
    performance = [0.71,0.79, 0.915, 0.803, LCS_Recall, GFS_Recall]
    # performance1 = [5, 15, 12, 45, 54]
    # plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.bar(y_pos, performance, bar_width, color='#375E97', edgecolor='black')
    # plt.bar(y_pos + bar_width, performance1, bar_width, color='#FB6542', edgecolor='black')

    plt.xticks(y_pos, objects, fontsize=10, rotation=0)
    plt.xlabel('Methods', fontsize=10)
    plt.ylabel('Recall', fontsize=10)
    plt.title('Recall Graph')
    # plt.legend(Classifiers, loc=2)
    plt.show()

def time_GFS(GFS_time):
    objects = ('FBR', 'WBorda', 'UDD', 'PEBL', 'LCS', 'GFS')
    y_pos = np.arange(len(objects))
    # Classifiers = ['KDD', 'NSL']
    bar_width = 0.70
    performance = [1.31,1.51, 0.85, 1.42, LCS_time, GFS_time]
    # performance1 = [5, 15, 12, 45, 54]
    # plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.bar(y_pos, performance, bar_width, color='#375E97', edgecolor='black')
    # plt.bar(y_pos + bar_width, performance1, bar_width, color='#FB6542', edgecolor='black')

    plt.xticks(y_pos, objects, fontsize=10, rotation=0)
    plt.xlabel('Methods', fontsize=10)
    plt.ylabel('Aveg Execution Time(S)', fontsize=10)
    plt.title('Time Graph')
    # plt.legend(Classifiers, loc=2)
    plt.show()

