import csv
from dataclasses import dataclass
import numpy as np
import pandas as pd
from skimage import measure
import inspect
import warnings
from scipy.spatial.distance import cdist
from tkinter import *
import math
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import cv2



class RQA:
    def main():
        x = []

        file = 'fnnData.csv'

        with open(file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                x.append(row.get('Row1'))

        def rqa_body(data, type, emb, delay, zscore, norm, setpara, setvalue):
            argspec = inspect.getargspec(rqa_body)
            argnum = len(argspec.args)

            if argnum < 7 or setpara == None:
                setpara = 'radius'
            setpara = setpara.lower()

            if argnum < 8 or setvalue == None:
                if setpara == 'radius' or 'rad' or 1:
                    radius = 1
                    runSetRad = 0
                elif setpara == 'perrec' or 'recurrence' or 2:
                    radiusStart = 0.01
                    radiusEnd = 0.5
                    runSetRad = 1
                    setvalue = 2.5

            else:
                if setpara == 'radius' or 'rad' or 1:
                    radius = setvalue
                    runSetRad = 0
                elif setpara == 'perrec' or 'recurrence' or 2:
                    radiusStart = 0.01
                    radiusEnd = 0.5
                    runSetRad = 1
            
            if argnum < 6 or norm == None:
                norm = 'non'
            if argnum < 5 or zscore == None:
                zscore = 0
            if argnum < 4 or delay == None:
                delay = 1
            if argnum < 3 or emb == None:
                emb = 1
            if argnum < 2 or type == None:
                type = 'RQA'
            
            if zscore:
                data = zscore(data)

            c = np.size(data,2)

            if type == 'RQA':
                dim = 1
                if c > 2:
                    data = data[:,1]
                    warnings.warn('More than one column of data. Only using first column.')
            elif type == 'CRQA' or 'CROSS':
                dim = 2
                if c > 2:
                    data = data[:,1:2]
                    warnings.warn('More than two columns of data. Only using first two columns.')
            elif type == 'JQRA' or 'JOINT':
                dim = c
                if c > 2:
                    data = data[:,1:2]
                    warnings.warn('Input data must have at least two columns.')
            elif type == 'MDRQA' or 'MD' or 'MULTI':
                dim = c

            tempData = []
            if emb > 1:
                for i in range(0, emb):
                    tempData[0:len(data)-(emb-1)*delay,1+dim *(i-1):dim*i] = data[1+(i-1)*delay:len(data)-(emb-i)*delay,:]
                data = tempData

            a = {}
            if type == 'RQA':
                print(5)
                a[1] = cdist(data,data)
                a[1] = abs(a[1])*-1

            #Change index??
            elif type == 'CRQA' or 'CROSS':
                a[1] = cdist(data[:,1:dim:-1], data[:,2:dim:-1]) 
                a[1] = abs(a[1])*-1
            
            elif type == 'JRQA' or 'JOINT':
                for i in range(0, c):
                    a[i] = cdist(data[:,i:dim-1], data[:,i:dim:-1])
                    a[i] = abs(a[i])*-1

            elif type == 'MDRQA' or 'MD' or 'MULTI':
                a[1] = cdist(data, data)
                a[1] = abs(a[1])*-1

            if norm.__contains__('euc'):
                print(5)
                for i in range(0, len(a)):
                    b = np.mean(a[i](a[i]<0))
                    b = -np.sqrt(abs((b^2)+2*(dim*emb)))
                    a[i] = a[i]/abs(b)
            
            elif norm.__contains__('min'):
                for i in range(0, len(a)):
                    b = max(a[i](a[i]<0))
                    a[i] = a[i]/abs(b)
            
            elif norm.__contains__('max'):
                for i in range(0, len(a)):
                    b = min(a[i](a[i]<0))
                    a[i] = a[i]/abs(b)
            else:
                # MAKE ERROR?
                print('No appropriate norm paramter specified')

            weight_r_plot = a
            for i in range(0, np.size(a,2)-1):
                weight_r_plot[i+1] = np.multiply(weight_r_plot[i], weight_r_plot[i+1])
            
            if i:
                weight_r_plot = -(abs(np.power(weight_r_plot[i], (1/(i+1)) )))

            if type(weight_r_plot) == list:
                weight_r_plot = weight_r_plot[1]

            if setpara == 'radius' or 'rad' or 1:
                [perRec, diag_hist, vertical_hist,A] = recurrenceMethod(a,radius)
            elif setpara == 'perrec' or 'recurrence' or 2:
                [perRec, diag_hist, vertical_hist,A] = recurrenceMethod(a,radiusStart)
            
            if perRec > 0:
                #                      Check if allowed
                DET = 100*sum(diag_hist[diag_hist>1]) / sum(diag_hist)
                MeanL = np.mean(diag_hist[diag_hist>1])
                MaxL = max(diag_hist)
                [count, bin] = plt.hist[diag_hist[diag_hist>1], min(diag_hist[diag_hist>1]) : max(diag_hist) ]
                total = sum(count)
                p = np.divide(count, total)
                deL = np.argwhere(count == 0)
                p[deL] = []
                EntrL = -sum(p * np.log2(p))
                
                LAM = 100*sum(vertical_hist[vertical_hist > 1]) / sum(vertical_hist)
                
                MeanVertical = np.mean(vertical_hist[vertical_hist > 1])
                MaxVertical = max(vertical_hist)
                [count, bin] = plt.hist[vertical_hist[vertical_hist>1], min(vertical_hist[vertical_hist>1]) :max(vertical_hist)]
                total = sum(count)
                p = np.divide(count, total)
                p[deL] = []
                EntrV = -sum(p * np.log2(p))
                EntrW = RQA_WeightedEntropy(weight_r_plot)
            else:
                DET = math.nan
                MeanL = math.nan
                MaxL = math.nan
                EntrL = math.nan
                LAM = math.nan
                MeanVertical = math.nan
                MaxVertical = math.nan
                EntrV = math.nan
                EntrW = math.nan
            RP = cv2.rotate(A, cv2.ROTATE_180)

            root = Tk()

            scrsz = root.winfo_screenwidth(), root.winfo_screenheight()
            f = Frame(root)
            f.pack(fill=BOTH, expand=YES)
            tabControl = Notebook(f)
            tab1 = Frame(tabControl)
            tabControl.add(tab1, text = "Binary")
            tab2 = Frame(tabControl)
            tabControl.add(tab2, text = "Heatmap")
            tabControl.pack(fill=BOTH, expand=YES)

            a1 = Frame(tab1)
            a1.place(relx = 0, rely = 0, relwidth = 1, relheight = 1)

            ax1 = plt.Axes(a1, [.375, .35, .58, .6])
            ax1.imshow(RP, cmap='gray')
            ax1.set_title('DIM = {}; EMB = {}; DEL = {}; RAD = {}; NORM = {}; ZSCORE = {}'.format(dim,emb,delay,setpara,norm,zscore), fontsize=8)
            ax1.set_xlabel('X(i)', fontsize=10)
            ax1.set_ylabel('Y(j)', fontsize=10)
            ax1.set_xticks([])
            ax1.set_yticks([])

            figure1 = Figure(figsize = (5, 4), dpi = 100)
            ax1 = figure1.add_subplot(111)

            if type.upper() in ['RQA','MDRQA','MD','MULTI']:
                ax2 = plt.Axes(tab1, [.375, .1, .58, .15])
                ax2.plot(range(len(data[:,1])), data[:,1], 'k-')
                ax2.set_xlim([1, len(data[:,1])])
                ax3 = plt.Axes(tab1, [.09, .35, .15, .6])
                ax3.plot(np.flip(data[:,1]), range(len(data[:,1])), 'k-')
                ax3.set_ylim([1, len(data[:,1])])
                ax3.invert_yaxis()
            elif type.upper() in ['CRQA','CROSS']:
                ax2 = plt.Axes(tab1, [.375, .1, .58, .15])
                ax2.plot(range(len(data[:,1])), data[:,1], 'k-')
                ax2.set_xlim([1, len(data[:,1])])
                ax3 = plt.Axes(tab1, [.09, .35, .15, .6])
                ax3.plot(np.flip(data[:,2]), range(len(data[:,2])), 'k-')
                ax3.set_ylim([1, len(data[:,1])])
            elif type.upper() in ['JRQA' 'JOINT']:
                for i in range(c):
                    ax2 = binary.add_axes([.375, .1, .58, .15], fontsize=8)
                    ax2.plot(range(len(data[:,0])), data[:,i], 'k-')
                    ax2.set_xlim([1, len(data[:,0])])
                    ax3 = binary.add_axes([.09, .35, .15, .6], fontsize=8)
                    ax3.plot(list(reversed(data[:,0])), range(len(data[:,i])), 'k-')
                    ax3.set_ylim([1, len(data[:,0])])
                    ax3.invert_yaxis()

            plt.sca(a1)
            str1 = ['%REC = ' + '{:.2f}'.format(RESULTS.REC)]
            plt.text(0.1, 0.27, str1, fontsize=8, color='k')
            str1 = ['%DET = ' + '{:.2f}'.format(RESULTS.DET)]
            plt.text(0.1, 0.24, str1, fontsize=8, color='k')
            str1 = ['MaxL = ' + '{:.0f}'.format(RESULTS.MaxL)]
            plt.text(0.1, 0.21, str1, fontsize=8, color='k')
            str1 = ['MeanL = ' + '{:.2f}'.format(RESULTS.MeanL)]
            plt.text(0.1, 0.18, str1, fontsize=8, color='k')
            str1 = ['EntrL = ' + '{:.2f}'.format(RESULTS.EntrL)]
            plt.text(0.1, 0.15, str1, fontsize=8, color='k')
            str1 = ['%LAM = ' + '{:.2f}'.format(RESULTS.LAM)]
            plt.text(0.1, 0.12, str1, fontsize=8, color='k')
            str1 = ['MaxV = ' + '{:.0f}'.format(RESULTS.MaxV)]
            plt.text(0.1, 0.09, str1, fontsize=8, color='k')
            str1 = ['MeanV = ' + '{:.2f}'.format(RESULTS.MeanV)]
            plt.text(0.1, 0.06, str1, fontsize=8, color='k')
            str1 = ['EntrV = ' + '{:.2f}'.format(RESULTS.EntrV)]
            plt.text(0.1, 0.03, str1, fontsize=8, color='k')


            heatmap = plt.figure()

            # Create first axis
            a2 = heatmap.add_axes([0, 0, 1, 1], visible=False)

            # Create second axis
            ax = [None] * 6
            ax[4] = heatmap.add_axes([0.375, 0.35, 0.58, 0.6], fontsize=8)
            wrp_rot = np.rot90(-1*weight_r_plot)
            ax[4].imshow(wrp_rot)
            ax[4].set_title('DIM = {}; EMB = {}; DEL = {}; RAD = {}; NORM = {}; ZSCORE = {}'.format(dim, emb, delay, radius, norm, zscore), fontsize=8)
            ax[4].set_xlabel('X(i)', fontsize=10)
            ax[4].set_ylabel('Y(j)', fontsize=10)
            ax[4].set_xticks([])
            ax[4].set_yticks([])

            # Create third, fourth, and fifth axes based on value of TYPE
            if type.upper() in ['RQA', 'MDRQA', 'MD', 'MULTI']:
                ax[5] = heatmap.add_axes([0.375, 0.1, 0.58, 0.15], fontsize=8)
                ax[5].plot(np.arange(len(data[:,0]))+1, data[:,0], 'k-')
                ax[5].set_xlim([1, len(data[:,0])])
                ax[6] = heatmap.add_axes([0.09, 0.35, 0.15, 0.6], fontsize=8)
                ax[6].plot(np.flip(data[:,0]), np.arange(len(data[:,0]))+1, 'k-')
                ax[6].set_ylim([1, len(data[:,0])])
                ax[6].invert_yaxis()
            elif type.upper() in ['CRQA', 'CROSS']:
                ax[5] = heatmap.add_axes([0.375, 0.1, 0.58, 0.15], fontsize=8)
                ax[5].plot(np.arange(len(data[:,0]))+1, data[:,0], 'k-')
                ax[5].set_xlim([1, len(data[:,0])])
                ax[6] = heatmap.add_axes([0.09, 0.35, 0.15, 0.6], fontsize=8)
                ax[6].plot(np.flip(data[:,1]), np.arange(len(data[:,1]))+1, 'k-')
                ax[6].set_ylim([1, len(data[:,0])])
                ax[6].invert_yaxis()
            elif type.upper() in ['JRQA','JOINT']:
                c = data.shape[1]
                for i in range(c):
                    ax[5] = heatmap.add_axes([0.375, 0.1, 0.58, 0.15], fontsize=8)
                    ax[5].plot(np.arange(len(data[:,0]))+1, data[:,i], 'k-')
                    ax[5].set_xlim([1, len(data[:,0])])
                    ax[6] = heatmap.add_axes([0.09, 0.35, 0.15, 0.6], fontsize=8)
                    ax[6].plot(np.flip(data[:,0]), np.arange(len(data[:,i]))+1, 'k-')
                    ax[6].set_ylim([1, len(data[:,0])])
                    ax[6].invert_yaxis()
            plt.sca(a2)
            str_list = [f"EntrW = {RESULTS['EntrW']:.2f}"]
            plt.text(.1, .27, "\n".join(str_list), fontsize=8, color='k')
            plt.link_axes([ax[0], ax[3]])
            plt.link_axes([ax[0], ax[1], ax[3], ax[4]])
            plt.link_axes([ax[0], ax[2], ax[3], ax[5]])
            plt.show()
                



                
                



            def setRadius(radius):
                [perRec, diag_hist, vertical_hist,A] = recurrenceMethod(a,radius)
                while perRec == 0 or perRec > 2.5:
                    print("Minimum redius has been adjusted")
                    if perRec == 0:
                        redius = radius*2
                    elif perRec > setvalue:
                        radius = radius / 1.5
                    [perRec, diag_hist, vertical_hist,A] = recurrenceMethod(a,radius)
                    while perRec < setvalue:
                        print("Meximum radius has been increased")
                        radiusEnd = radiusEnd *2
                        [perRec, diag_hist, vertical_hist,A] = recurrenceMethod(a,radiusEnd)
                    
                    lv = radius
                    hv = radiusEnd
                    target = setvalue
                    iter = 20
                    mid = [None]*20
                    rad = [None]*20
                    perRecIter = [None]*20
                    for i in range(0,iter):
                        mid[i] = (lv[i] + hv[i]) / 2
                        rad[i] = mid[i]
                        [perRec, diag_hist, vertical_hist,A] = recurrenceMethod(a,rad)

                        perRecIter[i] = perRec
                        if perRecIter[i] < target:
                            hv[i+1] = hv[i]
                            lv[i+1] = mid[i]
                        else:
                            lv[i+1] = lv[i]
                            hv[i+1] = mid[i]
                    perRecFinal = perRecIter[-1]
                    radiusFinal = rad[-1]
                    returnList = [perRec, diag_hist, vertical_hist, radiusFinal, A]
                    return returnList


        
            def recurrenceMethod(A, radius):
                for i in range(0, len(A)):
                    A[i] = A[i] + radius
                    #A[i](A[i] >= 0) = 1
                
                if len(A) > 1:
                    for i in range(0, len(A)-1):
                        A[i+1] = np.multiply(A[i], A[i+1])

                    A = A[i+1]
                else:
                    A = A[1]

                diag_hist = []
                vertical_hist = []

                for i in range( -(len(data)-1), len(data)-1):
                    print(1)
                    C = np.diag(A,i)
                    labels = measure.label(C, connectivity=8)
                    labels = pd.Series(labels)
                    frequency = labels.value_counts()
                    if frequency[0][0] == 0:
                        frequency = frequency[1:-1, 1]
                    else:
                        frequency = frequency[1]
                    
                    frequency = diag_hist[(len(diag_hist)+1) : (len(diag_hist) + len(frequency))]
                
                if type.__contains__('CROSS' or 'CRQA'):
                    perRec = 100*(sum (sum(A))-len(A)) / ((len(A))^2 - len(A))
                else:
                    perRec = 100*(sum (sum(A))) / ((len(A))^2)

                returnList = [perRec, diag_hist, vertical_hist, A]
                return returnList

            def RQA_WeightedEntropy(WRP):
                N = len(WRP)
                si = []
                for j in range(1, N):
                    print(1)
                    si[j] = sum(WRP[:,j])
                
                mi = min(si)
                ma = max(si)
                m = (ma-mi) / 49
                I = 1
                S = sum(si)
                p1 = []
                pp = []
                for s in range(mi, ma, m):
                    P = sum(si[si >= s and si < (s+m)])
                    p1[I] = P / S
                    I = I+1
                for I in range(0, len(p1)):
                    pp[I] = (p1[I]*np.log(p1[I]))
                
                pp[np.isnan(pp)] = 0
                Swrp = -1*(sum(pp))
                return Swrp
                

        


# root = Tk()

# scrsz = root.winfo_screenwidth(), root.winfo_screenheight()
# f = Frame(root)
# f.pack(fill=BOTH, expand=YES)
# tabControl = Notebook(f)
# tab1 = Frame(tabControl)
# tabControl.add(tab1, text = "Binary")
# tab2 = Frame(tabControl)
# tabControl.add(tab2, text = "Heatmap")
# tabControl.pack(fill=BOTH, expand=YES)

# a1 = Frame(tab1)
# a1.place(relx = 0, rely = 0, relwidth = 1, relheight = 1)

# ax1 = plt.Axes(a1, [.375, .35, .58, .6])
# ax1.imshow(RP, cmap='gray')
# ax1.set_title('DIM = {}; EMB = {}; DEL = {}; RAD = {}; NORM = {}; ZSCORE = {}'.format(DIM,EMB,DEL,RAD,NORM,ZSCORE), fontsize=8)
# ax1.set_xlabel('X(i)', fontsize=10)
# ax1.set_ylabel('Y(j)', fontsize=10)
# ax1.set_xticks([])
# ax1.set_yticks([])

# figure1 = Figure(figsize = (5, 4), dpi = 100)
# ax1 = figure1.add_subplot(111)

# if TYPE.upper() in ['RQA','MDRQA','MD','MULTI']:
#     ax2 = plt.Axes(tab1, [.375, .1, .58, .15])
#     ax2.plot(range(len(DATA[:,1])), DATA[:,1], 'k-')
#     ax2.set_xlim([1, len(DATA[:,1])])
#     ax3 = plt.Axes(tab1, [.09, .35, .15, .6])
#     ax3.plot(flip(DATA[:,1]), range(len(DATA[:,1])), 'k-')
#     ax3.set_ylim([1, len(DATA[:,1])])
#     ax3.invert_yaxis()
# elif TYPE.upper() in ['CRQA','CROSS']:
#     ax2 = plt.Axes(tab1, [.375, .1, .58, .15])
#     ax2.plot(range(len(DATA[:,1])), DATA[:,1], 'k-')
#     ax2.set_xlim([1, len(DATA[:,1])])
#     ax3 = plt.Axes(tab1, [.09, .35, .15, .6])
#     ax3.plot(flip(DATA[:,2]), range(len(DATA[:,2])), 'k-')
#     ax3.set_ylim([1, len(DATA[:,1])])
# elif case == 'JRQA' or case == 'JOINT':
#     for i in range(c):
#         ax2 = binary.add_axes([.375, .1, .58, .15], fontsize=8)
#         ax2.plot(range(len(DATA[:,0])), DATA[:,i], 'k-')
#         ax2.set_xlim([1, len(DATA[:,0])])
#         ax3 = binary.add_axes([.09, .35, .15, .6], fontsize=8)
#         ax3.plot(list(reversed(DATA[:,0])), range(len(DATA[:,i])), 'k-')
#         ax3.set_ylim([1, len(DATA[:,0])])
#         ax3.invert_yaxis()

# plt.sca(a1)
# str1 = ['%REC = ' + '{:.2f}'.format(RESULTS.REC)]
# plt.text(0.1, 0.27, str1, fontsize=8, color='k')
# str1 = ['%DET = ' + '{:.2f}'.format(RESULTS.DET)]
# plt.text(0.1, 0.24, str1, fontsize=8, color='k')
# str1 = ['MaxL = ' + '{:.0f}'.format(RESULTS.MaxL)]
# plt.text(0.1, 0.21, str1, fontsize=8, color='k')
# str1 = ['MeanL = ' + '{:.2f}'.format(RESULTS.MeanL)]
# plt.text(0.1, 0.18, str1, fontsize=8, color='k')
# str1 = ['EntrL = ' + '{:.2f}'.format(RESULTS.EntrL)]
# plt.text(0.1, 0.15, str1, fontsize=8, color='k')
# str1 = ['%LAM = ' + '{:.2f}'.format(RESULTS.LAM)]
# plt.text(0.1, 0.12, str1, fontsize=8, color='k')
# str1 = ['MaxV = ' + '{:.0f}'.format(RESULTS.MaxV)]
# plt.text(0.1, 0.09, str1, fontsize=8, color='k')
# str1 = ['MeanV = ' + '{:.2f}'.format(RESULTS.MeanV)]
# plt.text(0.1, 0.06, str1, fontsize=8, color='k')
# str1 = ['EntrV = ' + '{:.2f}'.format(RESULTS.EntrV)]
# plt.text(0.1, 0.03, str1, fontsize=8, color='k')


# heatmap = plt.figure()

# # Create first axis
# a2 = heatmap.add_axes([0, 0, 1, 1], visible=False)

# # Create second axis
# ax = [None] * 6
# ax[4] = heatmap.add_axes([0.375, 0.35, 0.58, 0.6], fontsize=8)
# wrp_rot = np.rot90(-1*wrp)
# ax[4].imshow(wrp_rot)
# ax[4].set_title('DIM = {}; EMB = {}; DEL = {}; RAD = {}; NORM = {}; ZSCORE = {}'.format(DIM, EMB, DEL, radius, NORM, ZSCORE), fontsize=8)
# ax[4].set_xlabel('X(i)', fontsize=10)
# ax[4].set_ylabel('Y(j)', fontsize=10)
# ax[4].set_xticks([])
# ax[4].set_yticks([])

# # Create third, fourth, and fifth axes based on value of TYPE
# if TYPE.upper() in ['RQA', 'MDRQA', 'MD', 'MULTI']:
#     ax[5] = heatmap.add_axes([0.375, 0.1, 0.58, 0.15], fontsize=8)
#     ax[5].plot(np.arange(len(DATA[:,0]))+1, DATA[:,0], 'k-')
#     ax[5].set_xlim([1, len(DATA[:,0])])
#     ax[6] = heatmap.add_axes([0.09, 0.35, 0.15, 0.6], fontsize=8)
#     ax[6].plot(np.flip(DATA[:,0]), np.arange(len(DATA[:,0]))+1, 'k-')
#     ax[6].set_ylim([1, len(DATA[:,0])])
#     ax[6].invert_yaxis()
# elif TYPE.upper() in ['CRQA', 'CROSS']:
#     ax[5] = heatmap.add_axes([0.375, 0.1, 0.58, 0.15], fontsize=8)
#     ax[5].plot(np.arange(len(DATA[:,0]))+1, DATA[:,0], 'k-')
#     ax[5].set_xlim([1, len(DATA[:,0])])
#     ax[6] = heatmap.add_axes([0.09, 0.35, 0.15, 0.6], fontsize=8)
#     ax[6].plot(np.flip(DATA[:,1]), np.arange(len(DATA[:,1]))+1, 'k-')
#     ax[6].set_ylim([1, len(DATA[:,0])])
#     ax[6].invert_yaxis()
# elif TYPE.upper() in ['JRQA','JOINT']:
#     c = DATA.shape[1]
#     for i in range(c):
#         ax[5] = heatmap.add_axes([0.375, 0.1, 0.58, 0.15], fontsize=8)
#         ax[5].plot(np.arange(len(DATA[:,0]))+1, DATA[:,i], 'k-')
#         ax[5].set_xlim([1, len(DATA[:,0])])
#         ax[6] = heatmap.add_axes([0.09, 0.35, 0.15, 0.6], fontsize=8)
#         ax[6].plot(np.flip(DATA[:,0]), np.arange(len(DATA[:,i]))+1, 'k-')
#         ax[6].set_ylim([1, len(DATA[:,0])])
#         ax[6].invert_yaxis()
# plt.sca(a2)
# str_list = [f"EntrW = {RESULTS['EntrW']:.2f}"]
# plt.text(.1, .27, "\n".join(str_list), fontsize=8, color='k')
# plt.link_axes([ax[0], ax[3]])
# plt.link_axes([ax[0], ax[1], ax[3], ax[4]])
# plt.link_axes([ax[0], ax[2], ax[3], ax[5]])
# plt.show()
    
    if __name__ == "__main__":
        main()