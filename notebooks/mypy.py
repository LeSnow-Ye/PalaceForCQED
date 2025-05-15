# -*- coding: Shift_JIS -*-
"""
modules by Kono
Created on Thu May 01 16:35:42 2014

@author: qipe
"""

#import scipy as S
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib as mpl
import os
import glob
import csv
import scipy.optimize as fit
from scipy import signal
from scipy.fftpack import fft
from scipy.signal import find_peaks_cwt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
from copy import deepcopy
'''
try:
    import Labber
except:
    print('No labber')
'''

'''
### marker size ###
params = mp.params
params['lines.markersize'] = 2
mpl.rcParams.update(params)

### Rastering for plot ###
ax.plot(..., rasterized=True)

for i, x in enumerate(xvec):
    print(i, x)

filename = 'fig' 
fig, ax = plt.subplots(1, 1,figsize=(figh/mp.inch,figv/mp.inch)) 
ax.plot(xf,yf,'r',label='')
ax.errorbar(x,y,yerr=yerr,fmt='bo',ecolor='k',capsize=1,alpha=0.5,label='label',rasterized=True)
ax.set_xlabel(r'x ()')
ax.set_ylabel(r'y ()')
#ax.set_xlim(0,1)
#ax.set_ylim(0,1)
#ax.set_xticks(np.linspace(0,1,5))
#ax.set_yticks(np.linspace(0,1,5))
#ax.legend()
#fig.savefig(filename+'.svg',dpi=200)
fig.show()    

'''

params = {
        'figure.autolayout': True,
        'figure.dpi': 200, 
        'pdf.fonttype': 42,
        'font.family': 'Arial', 
        'font.size': 7,
        'axes.linewidth' : 0.75,         
        'xtick.major.size':2,
        'ytick.major.size':2,    
        'xtick.direction': 'in', 
        'ytick.direction': 'in',
        'xtick.major.pad': 4,
        'ytick.major.pad': 2,              
        'xtick.labelsize' : 7,
        'ytick.labelsize' : 7,
        'xtick.major.width': 0.75,
        'ytick.major.width': 0.75,
        'xtick.bottom': True,
        'xtick.top': True,
        'ytick.left': True,
        'ytick.right': True,         
        'lines.linewidth': 0.75,  
        'lines.markersize': 3,              
        'lines.markeredgewidth': 0.0,
        'legend.frameon': True,
        'legend.numpoints': 1,
        'legend.loc': 'best',
        'legend.fontsize': 7,
        'axes.grid': False,
        'savefig.dpi': 300, 
          }

inch=2.54 #cm
figh=8 #cm 
figv=6 #cm

h=6.62607004 *10**(-34) #Plank
hbar=1.0545718 *10**(-34) #Plank/2pi
kB=1.38064852 *10**(-23) #Boltzmann
ec=1.60217662 *10**(-19) #charge
Phi0=2.067833831 *10**(-15) #magnetic quanta
phi0=2.067833831 *10**(-15)/2/np.pi # reduced magnetic quanta
c=2.99792458 *10**8 #light velocity
eps = 8.85418782 * 10**(-12) #vacuum  
mu = 1.25663706 * 10**(-6) #vacuum
ZQ = phi0/(2*ec)
### file management ###########################################################
    
def savecsv(a,name,dirname=None):
    if dirname==None:
        dirname=os.getcwd()
    arrayname=dirname+'/'+name+'.csv'
    np.savetxt(arrayname, a )

def loadcsv(name,dirname=None):
    if dirname==None:
        dirname=os.getcwd()
    arrayname=dirname+'/'+name+'.csv'
    print(arrayname)
    a = np.loadtxt(arrayname)   
    return a

def getname(dirs,filenum=None,datatype='Labber'):
    dirs0 = dirs.copy()
    curdir=os.getcwd()
    #print(curdir)
    os.chdir(dirs0.pop(0))
    # LabRAD
    if filenum is None:
        dirs0.append(None)
    elif datatype == 'Labrad':
        filenum = str(filenum)
        dirs0.append(['0'*(5-len(filenum))+filenum]+['csv'])
    elif datatype == 'Labber':
        filenum = str(filenum)
        dirs0.append(['0'*(5-len(filenum))+filenum]+['hdf5'])
        #dirs0.append([filenum]+['hdf5'])
    elif datatype == 'Comsol':
        dirs0.append([str(filenum)]+['csv'])
    else:
        raise ValueError(datatype+' cannot be opened. Plese check datatype.')
    for i in range(len(dirs0)):
        dir_i = dirs0[i]
        #print(i,dir_i)
        if dir_i is not None:
            if type(dir_i)==list:
                globdir= glob.glob('*'+str(dir_i[0])+'*')
            else:
                globdir= glob.glob('*'+str(dir_i)+'*')
                dir_i = [dir_i]
            #print(dir_i)
            #print(globdir)
            dir_remove=[]
            dir_i_dum =[]
            for d in dir_i:
                if d[0]=='!':
                    dir_remove.append(d[1:])
                else:
                    dir_i_dum.append(d)
            dir_i = dir_i_dum
            
            candidate = globdir
            for d in dir_i:
                if len(dir_remove) >= 1:
                    for dr in dir_remove:
                        #print('dr =',dr)
                        candidate = [s for s in candidate if (str(d) in s) and (str(dr) not in s)]
                else:
                    candidate = [s for s in candidate if (str(d) in s)]
                    
            #print(globdir)
            if len(candidate)==0:
                raise ValueError(dirs0[i], 'not in',os.listdir(path='.'))
            elif len(candidate)>1:
                raise ValueError(dirs0[i], 'cannot specify in',globdir)
            name = candidate[0]
            #print(i,name)
            if i != len(dirs0)-1:
                os.chdir(name)
    #print(dir_i)
    if dir_i is None:
        if datatype == 'Labrad':
            name= glob.glob('*'+'.csv')[-1]
        elif datatype == 'Labber':        
            name= glob.glob('*'+'.hdf5')[-1]
    print('filename:',name)
    output = os.path.join(os.getcwd(),name)
    os.chdir(curdir)
    return output
        
### Loaddata ##################################################################    
def loaddata(dirs,filenum=None,var_num=[0],data_num=[0,1],datatype='Labrad',**kwargs):
    name = getname(dirs,filenum=filenum,datatype=datatype)
    if datatype == 'Labrad':
        output = LabRADdata(name,var_num,data_num)
    elif datatype == 'Comsol':
        dmin=kwargs['dmin']
        dmax=kwargs['dmax']
        output = Comsoldata(name,var_num,data_num,dmin=dmin,dmax=dmax)
    return output

def LabRADdata(name,var_num,data_num,Raw=False):
    data = np.loadtxt(name, delimiter = ',')
    output =[]
    if np.size(var_num) ==1:
        output.append(data[:,var_num[0]])
        for i in data_num:
            output.append(data[:,i+1])
    elif np.size(var_num) ==2:
        num0  = len(np.unique(data[:,np.amin(var_num)]))
        num1  = len(set(data[:,np.amax(var_num)]))
        # print(num0,num1)
        output.append(np.reshape(data[:,var_num[0]], (num0, num1)))
        output.append(np.reshape(data[:,var_num[1]], (num0, num1)))
        for i in data_num:
            output.append(np.reshape(data[:,i+2], (num0, num1)))
        output=np.array(output)       
        if var_num[0]>var_num[1]:
            output = np.transpose(output,(0,2,1))
    else:
        raise ValueError('The varible number should be up to 2.')
    return output

def Labberdata(name,var_num,data_num):
    f = Labber.LogFile(name)
    output = []
    step_channels = f.getStepChannels()
    channel_names = []
    for channel in step_channels:
        #print(channel['name'])
        data_ch = f.getData(channel['name'])
        #print(np.array(list(set(data_ch.flatten()))))
        size_ch = len(set(data_ch.flatten()))
        #print(data_set.size)
        if size_ch>2:
            channel_names.append(channel['name'])
    #print(channel_names)
    var_dum = []
    #print(var_num)
    for name in var_num:
        # print(name)
        if type(name) is str:
            dum = []
            for ch_name in channel_names:
                if name in ch_name:
                    dum.append(ch_name)
            #print(dum)
            if len(dum)==0:
                raise ValueError(name, 'not in',channel_names)
            elif len(dum)>1:
                raise ValueError(name, 'cannot specify in',channel_names)
            else:
                var_dum.append(dum[0])
        else:
            print('x label =',channel_names[name])
            var_dum.append(channel_names[name])         
    var_num = var_dum
    #print(var_num)            
    if np.size(var_num)==1:
        if f.getData(var_num[0]).shape[0]==1:
            x = f.getData(var_num[0])[0]
            c = f.getData()[0]
        else:
            x = f.getData(var_num[0])[:,0]
            c = f.getData()[:,0]
            #print(x.shape)
        output.append(x)
        flag_transpose = False 
    else:
        diffs = []
        output = []
        for (i, name) in enumerate(var_num):
            x = f.getData(name)
            diffs.append((x.flatten()[1]-x.flatten()[0]))
            output.append(x)
        c = f.getData()
        #print(diffs)
        if diffs[0] < diffs[1]:
            flag_transpose = False
        else:
            flag_transpose = True       
    data_dum = []
    data_dum.append(np.absolute(c))
    data_dum.append(np.angle(c))
    data_dum.append(np.real(c))
    data_dum.append(np.imag(c))
    data_dum.append(c)
    if len(data_num)==0:
        output.append(data_dum[-1])
    else:
        for i in data_num:
            output.append(data_dum[i])
    #print(data.shape)
    if flag_transpose:
        output = np.transpose(output,(0,2,1))
    return output

def getValue(f,name):
    stepchannels = f.getStepChannels()
    names=[]
    #print(stepchannels)
    for i in stepchannels:
        if name in i["name"]:
            names.append(i["name"])
            output = i['values']
    if np.size(names)>1:
        raise ValueError(name, 'cannot specify in',names)
    return output

def Comsoldata(name,var_num=[0],data_num=[2],dmin=None,dmax=None):
    f = open(name, 'r')
    data=[]
    reader = csv.reader(f)
    # header = next(reader)
    for row in reader:
        #print row
        if row[0][0] != '%':
            rowdata=[]
            for i in var_num+data_num:
                rowdata.append(float(row[i]))
            data.append(rowdata)
    f.close()
    data=np.array(data)
    #print data
    datas=[]
    if np.size(var_num)==1:
        for num in range(np.size(data_num)):
            datan = []
            var_bare = data[:,0]
            var = np.sort(np.array(list(set(var_bare))))
            data_bare = data[:,num+1]
            datan.append(var)
            datai = []
            if num == 0:
                indi = []
            for i in range(np.size(var)):
                ivec= data_bare[np.where(var_bare==var[i])[0]]
                if num == 0:
                    if dmin is None:
                        vmin = np.amin(ivec)
                    else:
                        vmin = dmin
                    if dmax is None:
                        vmax = np.amax(ivec)
                    else:
                        vmax = dmax
                    indmin = list(np.where(ivec<vmin)[0])
                    indmax = list(np.where(ivec>vmax)[0])
                    indvec = indmin + indmax
                    indi.append(indvec)
                else:
                    indvec=indi[i]
                ivec = np.delete(ivec, indvec)
                datai.append(ivec)
            datan.append(np.array(datai))
            datas.append(datan)
    else:
        for num in range(np.size(data_num)):
            datan = []
            var0_bare = data[:,0]
            var0 = np.sort(np.array(list(set(var0_bare))))
            var1_bare = data[:,1]
            data_bare = data[:,num+2]
            datan.append(np.array(var0))
            datam =[]
            if num == 0:
                indi=[]
            for i in range(np.size(var0)):
                datai=[]
                var1i_bare = var1_bare[np.where(var0_bare==var0[i])[0]]
                datai_bare = data_bare[np.where(var0_bare==var0[i])[0]]
                var1i = np.sort(np.array(list(set(var1i_bare))))
                datai.append(np.array(var1i))
                dataj=[]
                if num == 0:
                    indj=[]
                for j in range(np.size(var1i)):
                    jvec=datai_bare[np.where(var1i_bare==var1i[j])[0]]
                    if num== 0:
                        if dmin is None:
                            vmin = np.amin(jvec)
                        else:
                            vmin = dmin
                        if dmax is None:
                            vmax = np.amax(jvec)
                        else:
                            vmax = dmax
                        indmin = list(np.where(jvec<vmin)[0])
                        indmax = list(np.where(jvec>vmax)[0])
                        indvec = indmin + indmax
                        indj.append(indvec)
                    else:
                        indvec=indi[i][j]
                    jvec = np.delete(jvec, indvec)
                    dataj.append(jvec)
                datai.append(np.array(dataj))
                datam.append(datai)
                if num == 0:
                    indi.append(indj)      
            datan.append(datam)
            datas.append(datan)
    return datas

### Array processing ##########################################################
def index(vec,valuevec):
    '''
    [Input]
    vec = array(1,n)
    valuevec = array(1,m) or value # value in vec 
    [Output]
    index = value
    indexvec = array(1,m) 
    
    you can get index of vec at valuevec
    ###    
    indexvec=mp.index(vec,valuevec)
    ###
    '''
    vec = np.array(vec)
    if np.iterable(valuevec) == 0:
        index = np.argmin(np.absolute(vec - valuevec))
        return index        
    else:
        valuevec = np.array(valuevec)
        indexvec = []
        for value in valuevec: 
            vindex = np.argmin(np.absolute(vec - value))
            indexvec.append(vindex)
        indexvec = np.array(indexvec)
        return indexvec


### Get parameters ######################################################
def digit_correct(value,err):
    if np.sum(err) == 0.0:
        value_c = value
        err_c = 0.0
    else:
        err_d = int(-1*np.floor(np.log10(err)))
        value_c = round(value,err_d)
        err_c = round(err,err_d)
    return value_c, err_c

def popt_correct(popt,perr,print_on=True):
    popt_c = []
    perr_c = []
    for i in range(popt.size):
        popti= popt[i]
        perri= perr[i]
        popti_c, perri_c = digit_correct(popti,perri)
        popt_c.append(popti_c)
        perr_c.append(perri_c)
        if print_on:
            print('popt['+str(i)+'] =',popti_c, '+-',perri_c)
    return popt_c,perr_c
    
### Data processing ###########################################################
def get_offset(yvec,bin_num=None):
    if bin_num is None:
        bin_num = int(np.size(yvec)/10)
    bins = np.linspace(np.amin(yvec), np.amax(yvec), bin_num)
    hist = np.histogram(yvec, bins=bins)[0]
    bins = 0.5*(bins[1:] + bins[:-1])
    mode_idx = np.argmax(hist)
    offsetv = bins[mode_idx]
    return offsetv
    
def reim(am,ph):
    z = am*np.exp(1j*ph)
    re = np.real(z)
    im = np.imag(z)
    return re,im

def amph(re,im):
    z = re+1j*im
    am = np.absolute(z)
    ph = np.angle(z)
    return am,ph

def smooth(y,N=4,Ws=0.05):
    bess1, bess2 = signal.bessel(N, Ws, "low")
    yfilt = signal.filtfilt(bess1, bess2, y)
    return yfilt

def pca(xv,yv,instance=False):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X = np.vstack([xv, yv]).T
    pca.fit(X)
    PCA(copy=True, n_components=2, whiten=False)
    Xd = pca.transform(X)
    if instance: 
        return pca
    else:
        return Xd.transpose()[0], Xd.transpose()[1]

def search_signal(xvec,yvec,y2vec=None,filt=True):
    if filt:
        yfilt = smooth(yvec)
    else:
        yfilt = yvec
    yoffset = get_offset(yvec)
    if y2vec is None:
        bare = np.absolute(yvec-yoffset)
        sig= np.absolute(yfilt-yoffset)
    else:
        if filt:
            y2filt = smooth(y2vec)
        else:
            y2filt = y2vec
        y2offset = get_offset(y2vec)
        zdiff = (yvec+1j*y2vec)-(yoffset+1j*y2offset)
        max_ind = np.argmax(np.absolute(zdiff))
        x_dif = np.real(zdiff[max_ind])
        y_dif = np.imag(zdiff[max_ind])
        if x_dif > 0: 
            theta = np.arctan(y_dif/x_dif)
        else: 
            theta = np.arctan(y_dif/x_dif) + np.pi           
        zfilt = (yfilt+1j*y2filt)-(yoffset+1j*y2offset)    
        bare = np.real(zfilt*np.exp(-1j*theta))        
        sig = np.real(zdiff*np.exp(-1j*theta))
    ind_max = np.argmax(sig)
    if 0:
        plt.figure()
        plt.plot(xvec,bare,'b.')
        plt.plot(xvec,sig,'r.-')
        plt.show()
    x_max = xvec[ind_max]
    return x_max, sig

def power_spectrum(tvec,yvec):
    dx = tvec[1]-tvec[0]
    N = tvec.size
    fvec = np.fft.fftfreq(N, dx)
    fvec = np.fft.fftshift(fvec)
    yfvec = fft(yvec)
    yfvec = np.fft.fftshift(yfvec)/float(N)
    pvec = np.absolute(yfvec)**2
    if 0:
        plt.figure()
        plt.plot(fvec,pvec,'b')
        plt.show()
    return fvec, pvec

def phase_correct(phase):
    pd = np.diff(phase)
    indp = np.where(pd>np.pi)
    indm = np.where(pd<-np.pi)
    #print indp,indm
    for ind in indp[0]:
        phase[ind+1:] =phase[ind+1:]-2*np.pi
    for ind in indm[0]:
        phase[ind+1:] =phase[ind+1:]+2*np.pi 
    return phase

def tau_correct(freq,phase,correct_output=False):
    tau = get_offset(np.diff(smooth(phase)))/(freq[1]-freq[0])/2./np.pi
    output = -axb(freq,2*np.pi*tau,0) + np.average(axb(freq,2*np.pi*tau,0))
    if correct_output:
        #return output
        return tau
    phase += output
    return phase

def sortvec(xvec,yvec):
    xvecs=[]
    yvecs=[]
    while np.size(xvec)>0:
        ind=np.argmin(xvec)
        xvecs.append(xvec[ind])
        yvecs.append(yvec[ind])
        xvec=np.delete(xvec,ind)
        yvec=np.delete(yvec,ind)
    xvec=np.array(xvecs)
    yvec=np.array(yvecs)
    return xvec,yvec

def interpolate(x,y,xn):
    import scipy.interpolate
    rp = scipy.interpolate.splrep(x, y, s=0)
    yn = scipy.interpolate.splev(xn, rp, der=0)
    return yn 

def get_peak(x,y,N=4):
    xf = np.linspace(x[0],x[-1],1001)
    w = np.polyfit(x,y,N)
    yfit = np.polyval(w,xf)
    yedge = (y[0]+y[-1])/2.
    ymin = np.amin(yfit)
    ymax = np.amax(yfit)
    if np.absolute(ymin-yedge)>np.absolute(ymax-yedge):
        peak = ymin
        peakpos = xf[np.argmin(yfit)]
    else:
        peak = ymax
        peakpos = xf[np.argmax(yfit)]   
    return xf,yfit,peakpos,peak

def get_zgze(am, ph):
    zvec = am*np.exp(1j*ph)
    zveco = zvec - zvec[0]
    ampo = np.absolute(zveco)
    maxind = np.argmax(ampo)
    zg = zvec[0]
    ze = zvec[maxind]
    return zg,ze
   
def get_zgze_v2(c):
    zg = np.min(c)
    ze = np.max(c)
    return zg, ze

def get_pop(am,ph,zg,ze):    
    zvec = am * np.exp(1j * ph) - zg
    x_dif = np.real(ze - zg)
    y_dif = np.imag(ze - zg)
    if x_dif > 0: 
        theta = np.arctan(y_dif / x_dif)
    else: 
        theta = np.arctan(y_dif / x_dif) + np.pi
    pop = np.real(zvec * np.exp(-1j*theta) / np.absolute(ze - zg))
    return pop
    
### Fitting function ####################################################
def axb(xvec, a, b):
    return a*xvec+b

def abc(xvec, a, b, c):
    return a*xvec**2+b*xvec+c

def abx(xvec, a, b):
    return a*np.absolute(xvec-b)

def cos(xvec, a, b,c):
    return a*np.cos(b*xvec)+c

def exp(xvec, a, b, c):
    return a*np.exp(b*xvec)+c

def log(xvec, a, b, c):
    return a*np.log(xvec + b) + c
            
def Lorentzian(xvec, x0, a, gamma, offset):
    if gamma>0.0:
        return a/(((xvec-x0)/(gamma/2))**2 + 1)+offset
    else:
        return 10**8*np.ones(np.size(xvec))
    
def Lorentzian_fano(xvec, x0, a, gamma, x, y, offset):
    if gamma>0.0:
        return np.abs(gamma*a/(gamma/2 - 1j*(xvec-x0)) + x + 1j*y)**2 + offset
    else:
        return 10**8*np.ones(np.size(xvec))

def LorentziandB(xvec, x0, dB, gamma, offset):
    if gamma>0.0:
        return 10.0*np.log10(10**(dB/10.)/(((xvec-x0)/(gamma/2.))**2 + 1.)+offset)
    else:
        return 10**8*np.ones(np.size(xvec))

def gaussian(xvec, x0, a, b, offset):
    if b>0.0:
        return a*np.exp(-(xvec-x0)**2/(2*b**2))+offset
    else:
        return 10**8*np.ones(np.size(xvec))
    
from scipy.special import voigt_profile  
def voigt_RBW(xvec, x0, a, gamma, offset, RBW): # in power
    sigma = RBW/np.sqrt(2*np.pi)
    gamma/=2.
    if gamma>0.0:
        return a*(np.pi*gamma)*(sigma*np.sqrt(2*np.pi))*voigt_profile(xvec-x0, sigma, gamma, out=None) + offset
    else:
        return 10**8*np.ones(np.size(xvec))

def sinusoid(xvec,a,freq,phi,offset):
    return  a*np.cos(2.*np.pi*freq*xvec-phi)+offset

def damped_sinusoid(xvec,a,gamma,freq,phi,offset):
    return  a*np.exp(gamma*xvec)*np.cos(2.*np.pi*freq*xvec-phi) + offset

def damped_sinusoid_beat(xvec,a,gamma,freq1,freq2,phi1,phi2,offset):
    return a*np.exp(gamma*xvec)*(np.cos(2.*np.pi*freq1*xvec-phi1)+np.cos(2.*np.pi*freq2*xvec-phi2))+offset

def S11(x,f0,kex,kin):
    return ((kex-kin)/2-1j*(x-f0))/((kex+kin)/2+1j*(x-f0))

def S11exp_no_fano(x,f0,kex,kin,phi,tau):
    if (kex>=0) and (kin>=0):
        return np.exp(1j*phi+1j*2.*np.pi*x*tau)*((kex-kin)/2-1j*(x-f0))/((kex+kin)/2+1j*(x-f0))
    else:
        return 10**8*np.ones(np.size(x))

def S11exp(x,f0,kex,kin,phi,tau,x0,y0):
    if (kex>=0) and (kin>=0):
        return np.exp(1j*phi+1j*2.*np.pi*x*tau)*((kex-kin)/2-1j*(x-f0))/((kex+kin)/2+1j*(x-f0))+x0+1j*y0
    else:
        return 10**8*np.ones(np.size(x))

def S11exp_no_fano_with_amp(x,f0,kex,kin,phi,tau,amp):
    if (kex>=0) and (kin>=0):
        return amp*np.exp(1j*phi+1j*2.*np.pi*x*tau)*((kex-kin)/2-1j*(x-f0))/((kex+kin)/2+1j*(x-f0))
    else:
        return 10**8*np.ones(np.size(x))

def S11Qexp(x,f0,gex,gphi,phi,tau,x0,y0):
    if (gex>=0) and (gphi>=0):
        return np.exp(1j*phi+1j*2.*np.pi*x*tau)*(1-gex/(gphi+1j*(x-f0)))+x0+1j*y0
    else:
        return 10**8*np.ones(np.size(x))

### Automatically fitting #################################################### 
def axb_fit(xvec,yvec,sigma=None,ini=False):
    '''
    ###
    popt,pcov=mp.axb_fit(xvec,yvec)
    ###    
    '''
    if type(ini) is list:
        initial = ini      
    else:
        a= (yvec[-1]-yvec[0])/(xvec[-1]-xvec[0])
        b= yvec[0]-a*xvec[0]    
        initial=[a,b]
    popt, pcov = fit.curve_fit(axb, xvec, yvec,sigma=sigma, p0=initial)
    #print(popt)
    yfit = axb(xvec,*popt)
    yini = axb(xvec,*initial)
    if type(pcov)!=np.ndarray:
        pcov = np.zeros(np.size(popt)) 
    perr = np.sqrt(np.diag(pcov))
    return popt, perr, yfit, yini

 
def abc_fit(xvec,yvec):
    '''
    ###
    popt,pcov=mp.axb_fit(xvec,yvec)
    ###    
    '''
    popt, pcov = fit.curve_fit(abc, xvec, yvec)
    # print(popt)
    initial = [1.,1.,1.]
    yfit = abc(xvec,*popt)
    yini = abc(xvec,*initial)       
    if type(pcov)!=np.ndarray:
        pcov = np.zeros(np.size(popt))     
    perr = np.sqrt(np.diag(pcov))
    return popt, perr, yfit, yini

def ac_fit(xvec,yvec):
    '''
    ###
    popt,pcov=mp.ac_fit(xvec,yvec)
    ###    
    '''
    ac = lambda x, a, c: abc(x, a, 0, c)
    popt, pcov = fit.curve_fit(ac, xvec, yvec)
    # print(popt)
    initial = [1.,1.]
    yfit = ac(xvec,*popt)
    yini = ac(xvec,*initial)       
    if type(pcov)!=np.ndarray:
        pcov = np.zeros(np.size(popt))     
    perr = np.sqrt(np.diag(pcov))
    return popt, perr, yfit, yini
     
def abx_fit(xvec,yvec,ini=False):
    '''
    ###
    popt,pcov=mp.axb_fit(xvec,yvec)
    ###    
    '''
    if type(ini) is list:
        initial = ini
    else:
        minind = np.argmin(yvec)
        a= (yvec[-1]-yvec[minind])/(xvec[-1]-xvec[minind])
        b= xvec[minind]    
        initial=[a,b]
    popt, pcov = fit.curve_fit(abx, xvec, yvec, p0=initial)
    # print(popt)
    yfit = abx(xvec,*popt)
    yini = abx(xvec,*initial) 
    if type(pcov)!=np.ndarray:
        pcov = np.zeros(np.size(popt))     
    perr = np.sqrt(np.diag(pcov))  
    return popt, perr, yfit, yini

def cos_fit(xvec,yvec,ini=False):
    '''
    ###
    popt,pcov=mp.axb_fit(xvec,yvec)
    ###    
    '''
    if type(ini) is list:
        initial = ini
    else:
        a= (np.amax(yvec)-np.amin(yvec))/2.
        period = 2.*np.absolute(xvec[np.argmin(yvec)]-xvec[np.argmax(yvec)])
        b = 2.*np.pi/period
        c = (np.amax(yvec)+np.amin(yvec))/2. 
        initial=[np.sign(yvec[0]-c)*a,b,c]
    popt, pcov = fit.curve_fit(cos, xvec, yvec, p0=initial)
    # print(popt)
    yfit = cos(xvec,*popt)
    yini = cos(xvec,*initial) 
    if type(pcov)!=np.ndarray:
        pcov = np.zeros(np.size(popt))     
    perr = np.sqrt(np.diag(pcov))  
    return popt, perr, yfit, yini


def exp_fit(xvec,yvec,sigma=None,ini=False,bounds=False):
    '''
    ###
    popt,pcov=mp.exp_fit(xvec,yvec)
    ###    
    '''
    if type(ini) is list:
        initial = ini
    else:
        num = int(np.size(xvec)/10)+3
        popt1, perr1, yfit1, yini1=axb_fit(xvec[0:num],yvec[0:num])
        popt2, perr2, yfit2, yini2=axb_fit(xvec[-num:],yvec[-num:])
        if np.absolute(popt1[0]) > np.absolute(popt2[0]):       
            c= yvec[-1]
            a= yvec[0]-c
            b= popt1[0]/a
            initial=[a,b,c]
        if np.absolute(popt1[0]) <= np.absolute(popt2[0]):       
            c= yvec[0]
            a= yvec[-1]-c
            b= popt1[0]/a
            initial=[a,b,c]
            
    if type(bounds) is tuple:
        bounds = bounds 
        popt, pcov = fit.curve_fit(exp, xvec, yvec, sigma=sigma, p0=initial,bounds=bounds) 
    else:
        popt, pcov = fit.curve_fit(exp, xvec, yvec, sigma=sigma, p0=initial)

    # print('pcov',pcov)

    # print(popt)
    yfit = exp(xvec,*popt)
    yini = exp(xvec,*initial) 
    if type(pcov)!=np.ndarray:
        print('!!!!!! pcov allert in mp.exp_fit !!!!!!')
        pcov = np.zeroes(np.size(popt))
        # pcov = np.ones(np.size(popt))*np.inf     
    perr = np.sqrt(np.diag(pcov))  
    return popt, perr, yfit, yini

    
def damped_sinusoid_fit(xvec,yvec,ini=False):
    '''
    ###
    popt,pcov=mp.rabi_fit(xvec,yvec)
    ###    
    '''
    if type(ini) is list:
        initial = ini
    else:
        N = xvec.size
        fr,ps = power_spectrum(xvec,yvec)
        fr_pos = fr[N//2+1:]
        ps_pos = ps[N//2+1:]        
        freq = fr_pos[np.argmax(ps_pos)]
        f_int = np.linspace(fr_pos[0],fr_pos[-1],10001)
        p_int = interpolate(fr_pos,ps_pos,f_int)
        ind = index(p_int,np.amax(ps_pos)/2)
        gamma = -2*np.pi*np.absolute(freq - f_int[ind])
        if 0:
            plt.figure()
            plt.plot(fr_pos,ps_pos,'.',label=str(gamma))
            plt.plot(f_int,p_int,'-')
            plt.legend()
            plt.show() 
        offset = np.average(yvec)
        x_max = xvec[np.argmax(yvec)]
        phi = 2.*np.pi*freq*x_max 
        a = np.absolute(np.amax(yvec)-offset)*np.exp(-gamma*x_max) 
        initial = [a,gamma,freq,phi,offset]
    popt, pcov = fit.curve_fit(damped_sinusoid, xvec, yvec, p0=initial)
    # print popt,pcov
    yfit = damped_sinusoid(xvec,*popt)
    yini = damped_sinusoid(xvec,*initial)
    # print(pcov)
    if type(pcov)!=np.ndarray:
        pcov = np.zeros(np.size(popt)) 
    perr = np.sqrt(np.diag(pcov))
    return popt, perr, yfit, yini

def damped_sinusoid_beat_fit(xvec,yvec,ini=False):
    '''
    ###
    popt,pcov=mp.rabi_fit(xvec,yvec)
    ###    
    '''
    if type(ini) is list:
        initial = ini
    else:
        N = xvec.size
        fr,ps = power_spectrum(xvec,yvec)
        fr_pos = fr[N//2+1:]
        ps_pos = ps[N//2+1:]
        ps_pos_diff = np.diff(ps_pos)
        #plt.plot(ps_pos_diff)
        #idx = np.where(ps_pos_diff[:-1]*ps_pos_diff[1:]<np.absolute(np.argmax(ps_pos_diff))**2/4)[0]+1   
        idx = find_peaks_cwt(ps_pos,1.0+np.zeros(N))
        print(idx)     
        freq1 = fr_pos[idx[0]]
        #print(freq1)
        freq2 = fr_pos[idx[1]]
        #print(freq2)
        f_int = np.linspace(fr_pos[0],fr_pos[-1],10001)
        p_int = interpolate(fr_pos,ps_pos,f_int)
        ind1 = index(p_int,ps_pos[idx[0]]/2)
        ind2 = index(p_int,ps_pos[idx[1]]/2)
        gamma = -2*np.pi*(np.absolute(freq1 - f_int[ind1])+np.absolute(freq2 - f_int[ind2]))/2
        if 0:
            plt.figure()
            plt.plot(fr_pos,ps_pos,'.',label=str(gamma))
            plt.plot(f_int,p_int,'-')
            plt.legend()
            plt.show() 
        offset = np.average(yvec)
        yvec_diff = np.diff(yvec)
        idmax = np.where(yvec_diff[:-1]*yvec_diff[1:]<0)[0][0]
        #print(idmax)
        x_max = xvec[idmax]
        #print(x_max)
        phi_sum = -2.*np.pi*(freq1+freq2)*x_max
        phi_diff = -2.*np.pi*(freq1-freq2)*x_max
        phi1 = phi_sum+phi_diff
        phi2 = phi_sum-phi_diff
        a = np.absolute(np.amax(yvec)-offset)*np.exp(-gamma*x_max) 
        initial = [a,gamma,freq1,freq2,phi1,phi2,offset]
    popt, pcov = fit.curve_fit(damped_sinusoid_beat, xvec, yvec, p0=initial)
    # print popt,pcov
    yfit = damped_sinusoid_beat(xvec,*popt)
    yini = damped_sinusoid_beat(xvec,*initial)
    # print(pcov)
    if type(pcov)!=np.ndarray:
        pcov = np.zeros(np.size(popt)) 
    perr = np.sqrt(np.diag(pcov))
    return popt, perr, yfit, yini

def sinusoid_fit(xvec,yvec,ini=False):
    '''
    ###
    popt,pcov=mp.rabi_fit(xvec,yvec)
    ###    
    '''
    if type(ini) is list:
        initial = ini
    else:
        N = xvec.size
        fr,ps = power_spectrum(xvec,yvec)
        fr_pos = fr[N//2+1:]
        ps_pos = ps[N//2+1:]
        try:
            popt, perr, yfit, yini = Lorentzian_fit(fr_pos,ps_pos)
            freq = popt[0]
        except:
            if 0:
                plt.figure()
                plt.plot(fr_pos,ps_pos,'.')
                plt.show()        
            freq = fr_pos[np.argmax(ps_pos)]
        offset = np.average(yvec[N//2:])
        x_max = xvec[np.argmax(yvec)]
        phi = 2.*np.pi*freq*x_max 
        a = np.absolute(np.amax(yvec)-offset)
        initial = [a,freq,phi,offset]
    popt, pcov = fit.curve_fit(sinusoid, xvec, yvec, p0=initial)
    # print popt,pcov
    yfit = sinusoid(xvec,*popt)
    yini = sinusoid(xvec,*initial)
    # print(pcov)
    if type(pcov)!=np.ndarray:
        pcov = np.zeros(np.size(popt)) 
    perr = np.sqrt(np.diag(pcov))
    return popt, perr, yfit, yini

def gaussian_fit(xvec,yvec,ini=False): 
    '''
    popt = [x0, a, b, offset] #
    ###
    popt,pcov=mp.gaussian_fit(xvec,yvec)
    ###    
    '''
    if type(ini) is list:
        initial = ini
    else:
        offset = get_offset(yvec)
        ind_max = np.argmax(np.absolute(yvec-offset))
        idx_hwhm = np.argmin(np.absolute(yvec - (yvec[ind_max] + offset)/2.0))
        gamma = 2.0 * np.absolute(xvec[ind_max]-xvec[idx_hwhm])
        a = yvec[ind_max] - offset
        b = gamma/(2*np.sqrt(2*np.log(2)))
        initial=[xvec[ind_max],a,b,offset]
    popt, pcov = fit.curve_fit(gaussian, xvec, yvec, p0=initial)
    # popt[0] = popt[0] + x0
    yfit = gaussian(xvec,*popt)
    yini = gaussian(xvec,*initial)
    if type(pcov)!=np.ndarray:
        pcov = np.zeros(np.size(popt)) 
    perr = np.sqrt(np.diag(pcov))
    return popt, perr, yfit, yini


def voigt_fixed_RBW_fit(xvec,yvec,RBW,ini=False): 
    '''
    popt = [x0, a, gamma, offset, RBW]
    ###
    popt,pcov=mp.Lorentzian_fit(xvec,yvec)
    ###    
    '''
    
    def voigt_fixed_RBW(xvec, x0, a, gamma, offset):
        return voigt_RBW(xvec, x0, a, gamma, offset, RBW)
    
    if type(ini) is list:
        initial = ini       
    else:
        offset = get_offset(yvec)
        ind_max = np.argmax(np.absolute(yvec-offset))
        idx_hwhm = np.argmin(np.absolute(yvec - (yvec[ind_max] + offset)/2.0))
        gamma = 2.0 * np.absolute(xvec[ind_max]-xvec[idx_hwhm])
        a = yvec[ind_max] - offset
        initial=[xvec[ind_max],a,gamma,offset]
#         bounds = ([-np.inf,-np.inf,-np.inf,sigma],[np.inf,np.inf,np.inf,sigma])
    popt, pcov = fit.curve_fit(voigt_fixed_RBW, xvec, yvec, p0=initial)#, bounds=bounds)
    # popt[0] = popt[0] + x0
    yfit = voigt_fixed_RBW(xvec,*popt)
    yini = voigt_fixed_RBW(xvec,*initial)
    if type(pcov)!=np.ndarray:
        pcov = np.zeros(np.size(popt)) 
    perr = np.sqrt(np.diag(pcov))
    return popt, perr, yfit, yini

def Lorentzian_fit(xvec,yvec,ini=False): 
    '''
    popt = [x0, a, gamma, offset]
    ###
    popt,pcov=mp.Lorentzian_fit(xvec,yvec)
    ###    
    '''
    if type(ini) is list:
        initial = ini       
    else:
        offset = get_offset(yvec)
        ind_max = np.argmax(np.absolute(yvec-offset))
        idx_hwhm = np.argmin(np.absolute(yvec - (yvec[ind_max] + offset)/2.0))
        gamma = 2.0 * np.absolute(xvec[ind_max]-xvec[idx_hwhm])
        a = yvec[ind_max] - offset
        initial=[xvec[ind_max],a,gamma,offset]
    popt, pcov = fit.curve_fit(Lorentzian, xvec, yvec, p0=initial)
    # popt[0] = popt[0] + x0
    yfit = Lorentzian(xvec,*popt)
    yini = Lorentzian(xvec,*initial)
    if type(pcov)!=np.ndarray:
        pcov = np.zeros(np.size(popt)) 
    perr = np.sqrt(np.diag(pcov))
    return popt, perr, yfit, yini

def Lorentzian_fano_fit(xvec,yvec,ini=False): 
    '''
    popt = [x0, a, gamma, offset]
    ###
    popt,pcov=mp.Lorentzian_fit(xvec,yvec)
    ###    
    '''
    # first roughly fit a Lorentzian
    if type(ini) is list:
        initial = ini       
    else:
        offset = get_offset(yvec)
        ind_max = np.argmax(np.absolute(yvec-offset))
        idx_hwhm = np.argmin(np.absolute(yvec - (yvec[ind_max] + offset)/2.0))
        gamma = 2.0 * np.absolute(xvec[ind_max]-xvec[idx_hwhm])
        a = yvec[ind_max] - offset
        initial=[xvec[ind_max],a,gamma,offset]
    popt, pcov = fit.curve_fit(Lorentzian, xvec, yvec, p0=initial)
    
    # then fit a Fano Lorentzian
    initial_fano = [popt[0],np.sqrt(popt[1]),popt[2],0,0,popt[3]]
    popt, pcov = fit.curve_fit(Lorentzian_fano, xvec, yvec, p0=initial_fano)
    # popt[0] = popt[0] + x0
    yfit = Lorentzian_fano(xvec,*popt)
    yini = Lorentzian_fano(xvec,*initial_fano)
    if type(pcov)!=np.ndarray:
        pcov = np.zeros(np.size(popt)) 
    perr = np.sqrt(np.diag(pcov))
    return popt, perr, yfit, yini

def S11exp_fit(xvec,yvec,y2vec,ini=False): ###f0,kex,kin,phi,tau,x0,y0
    def residuals(params, x, y,y2):
            return np.absolute(y*np.exp(1j*y2) - S11exp(x, *params))
    #delta = np.amax(xvec)-np.amin(xvec)
    if type(ini) is list:
        f0 = ini[0]
        xvec -= f0
        #xvec = xvec/delta
        initial = ini
        initial[0] = 0.0
        #initial[1] = initial[1]/delta
        #initial[2] = initial[2]/delta
        initial[3] = initial[3] + 2.*np.pi*initial[4]*f0 #/delta
        #initial[4] = initial[4]/delta
    else:
        re,im = reim(yvec,y2vec)
        x_max,sig = search_signal(xvec,re,im,filt=False)
        popt0, perr0, yfit0, yini0 = Lorentzian_fit(xvec,sig)
        
        if 0:
            plt.figure()
            plt.plot(xvec,sig,'b.')
            plt.plot(xvec,yfit0,'r')
            plt.show()            
        
        f0 = popt0[0]
        # print(x0)
        xvec -= f0
        #xvec = xvec/delta
        amp = popt0[1]
        lw = popt0[2]#/delta
        kex = np.absolute(lw*(amp/2.))
        kin = np.absolute(lw*(1-amp/2.))
        tau = get_offset(np.diff(smooth(y2vec)))/(xvec[1]-xvec[0])/2./np.pi
        # phi = -np.pi + np.average(y2vec) - 2.*np.pi*xvec[0]*tau
        x0 = 0
        y0 = 0
        initial = [0.0,kex,kin,0.0,tau,x0,y0]
        phi =  np.average(np.unwrap(y2vec) - np.unwrap(np.angle(S11exp(xvec, *initial))))
        if 0:
            plt.figure()
            plt.plot(xvec,y2vec,'b.')
            plt.plot(xvec,2.*np.pi*tau*xvec, color="g")
            plt.plot(xvec,np.angle(S11exp(xvec, *initial)), color="r")
            plt.plot()
            plt.show()        
        initial = [0.0,kex,kin,phi,tau,x0,y0]
    popt,pcov,infodict,mesg,ier = fit.leastsq(residuals, initial, args=(xvec,yvec,y2vec),full_output=True)

    # print(pcov.shape)
    # print(pcov)

    ###
    xvec += f0 
    #xvec = xvec*delta + f0
    initial[0] = initial[0] + f0
    #initial[1] = initial[1]*delta
    #initial[2] = initial[2]*delta
    initial[3] = initial[3] - 2.*np.pi*initial[4]*f0 #/delta
    #initial[4] = initial[4]/delta
    popt[0] = popt[0] + f0
    #popt[1] = popt[1]*delta
    #popt[2] = popt[2]*delta
    popt[3] = popt[3] - 2.*np.pi*popt[4]*f0 #/delta
    #popt[4] = popt[4]/delta
    ###
    var = np.sum((np.absolute(yvec*np.exp(1j*y2vec) - S11exp(xvec, *popt)))**2)/(xvec.size-popt.size) 
    if type(pcov)!=np.ndarray:
        # Experimental Andrea and Xuxin feature:
        print('WARNING: Could not estimate covariance matrix. Fitting without Fano')
        return S11exp_no_fano_fit(xvec,yvec,y2vec,ini=False)
        # pcov = np.zeros((np.size(popt),np.size(popt)))    
    perr = np.sqrt(np.diag(pcov)*var)
    #perr[0] = perr[0]*delta 
    #perr[1] = perr[1]*delta
    #perr[2] = perr[2]*delta
    #perr[3] = np.sqrt(perr[3]**2+(2.*np.pi*popt[4]*popt[5]/delta)**2)
    #perr[4] = perr[4]/delta
    ###
    zfit = S11exp(xvec, *popt)
    zini = S11exp(xvec, *initial)
    return popt,perr,zfit,zini

def S11exp_fit_no_fano_with_amp(xvec,yvec,y2vec,ini=False): ###f0,kex,kin,phi,tau,amp
    def residuals(params, x, y,y2):
            return np.absolute(y*np.exp(1j*y2) - S11exp_no_fano_with_amp(x, *params))
    #delta = np.amax(xvec)-np.amin(xvec)

    if type(ini) is list:
        f0 = ini[0]
        xvec -= f0
        #xvec = xvec/delta
        initial = ini
        initial[0] = 0.0

        if np.average(np.diff(y2vec)/np.diff(xvec)) > 0 and initial[3] < np.pi/2:
            initial[3] += np.pi
        #initial[1] = initial[1]/delta
        #initial[2] = initial[2]/delta
        initial[3] += 2.*np.pi*initial[4]*f0 #/delta
        #initial[4] = initial[4]/delta

        re, im = reim(yvec,y2vec)
        z = (re+1j*im)
        tau = fit_delay(xvec, z, smoothing_sigma=5, nbr_sigma_trim=1)
    
        if np.abs(tau) < 1e-14:
            tau = get_offset(np.diff(smooth(y2vec)))/(xvec[1]-xvec[0])/2./np.pi
        initial[4] = tau
        # initial[3] = 0.0

        # phi = -np.pi + np.average(y2vec) - 2.*np.pi*xvec[0]*tau
        phi =  np.average(y2vec - np.angle(S11exp_no_fano_with_amp(xvec, *initial))) 
        if np.average(np.diff(y2vec)/np.diff(xvec)) > 0:
            phi += np.pi
        initial[3] = phi

    else:
        ref_lvl = get_offset(yvec)

        re,im = reim(yvec,y2vec)
        z = (re+1j*im)

        x_max,sig = search_signal(xvec,re,im,filt=False)
        popt0, perr0, yfit0, yini0 = Lorentzian_fit(xvec,sig)
        
        if 0:
            plt.figure()
            plt.plot(xvec,sig,'b.')
            plt.plot(xvec,yfit0,'r')
            plt.show()            
        
        f0 = popt0[0]
        # print(x0)
        xvec -= f0
        #xvec = xvec/delta
        amp = popt0[1] / ref_lvl
        lw = popt0[2]#/delta
        kex = np.absolute(lw*(amp/2.))
        kin = np.absolute(lw*(1-amp/2.))


        tau = fit_delay(xvec, z, smoothing_sigma=5, nbr_sigma_trim=1)
        if np.abs(tau) < 1e-14:
            tau = get_offset(np.diff(smooth(y2vec)))/(xvec[1]-xvec[0])/2./np.pi

        if 0:
            fig, ax = plt.subplots(figsize=(6*2/2.54, 4*2/2.54))
            z_temp1 = z * np.exp(1j*(2 * np.pi * tau * xvec))
            r0, xc, yc = circle_fit(z_temp1)
            ax.plot(r0*np.cos(np.linspace(-np.pi, np.pi, 100))+xc, r0*np.sin(np.linspace(-np.pi, np.pi, 100))+yc, 'g')
            ax.plot(np.real(z_temp1), np.imag(z_temp1), 'g.')
            ax.set_aspect('equal', adjustable='box')
            plt.tight_layout()
            print(tau)



        initial = [0.0,kex,kin,0.0,tau,ref_lvl]
        phi =  np.average(np.unwrap(y2vec) - np.unwrap(np.angle(S11exp_no_fano_with_amp(xvec, *initial))))

        if 0:
            plt.figure()
            plt.plot(xvec,y2vec,'b.')
            # plt.plot(xvec,2.*np.pi*tau*xvec)
            plt.plot(xvec,y2vec-np.angle(S11exp_no_fano_with_amp(xvec, *initial)))
            plt.plot(xvec,np.angle(S11exp_no_fano_with_amp(xvec, *initial)))
            initial2 = [0.0,kin,kex,0.0,tau,amp]
            plt.plot(xvec,np.angle(S11exp_no_fano_with_amp(xvec, *initial)))
            plt.show()

        initial = [0.0,kex,kin,phi,tau,ref_lvl]

    if 0:
        # correct potential electrical delay before
        re, im = reim(yvec,y2vec)
        z = (re+1j*im)

        popt0, perr0, yfit0, yin02 = Lorentzian_fit(xvec,np.abs(z)**2)
        if 0:
            plt.figure()
            plt.plot(xvec, np.abs(z)**2, 'b.')
            plt.plot(xvec, yfit0, 'b')
            plt.show()
        
        f0 = popt0[0]
        xvec -= f0
        amp = np.sqrt(popt0[3])
        if popt0[3] < np.abs(popt0[1]):
            ratio = popt0[3] / np.abs(popt0[3])
        else:
            ratio = popt0[1]/(amp**2)
        # print(ratio)
        # print(popt0)
        lw = popt0[2] #/delta
        kex = np.abs(lw/2.*(1+np.sqrt(1+ratio)))
        kin = np.abs(lw/2.*(1-np.sqrt(1+ratio)))

        tau = fit_delay(xvec, z, smoothing_sigma=5, nbr_sigma_trim=1)
    
        if np.abs(tau) < 1e-14:
            tau = get_offset(np.diff(smooth(y2vec)))/(xvec[1]-xvec[0])/2./np.pi

        if 0:
            fig, ax = plt.subplots(figsize=(6*2/2.54, 4*2/2.54))
            z_temp1 = z * np.exp(1j*(2 * np.pi * tau * xvec))
            r0, xc, yc = circle_fit(z_temp1)
            ax.plot(r0*np.cos(np.linspace(-np.pi, np.pi, 100))+xc, r0*np.sin(np.linspace(-np.pi, np.pi, 100))+yc, 'g')
            ax.plot(np.real(z_temp1), np.imag(z_temp1), 'g.')
            ax.set_aspect('equal', adjustable='box')
            plt.tight_layout()
            print(tau)



        initial = [0.0,kex,kin,0.0,tau,amp]
        # phi = -np.pi + np.average(y2vec) - 2.*np.pi*xvec[0]*tau
        phi =  np.average(y2vec - np.angle(S11exp_no_fano_with_amp(xvec, *initial))) 
        if np.average(np.diff(y2vec)/np.diff(xvec)) > 0:
            phi += np.pi

        if 0:
            plt.figure()
            plt.plot(xvec,y2vec,'b.')
            # plt.plot(xvec,2.*np.pi*tau*xvec)
            plt.plot(xvec,y2vec-np.angle(S11exp_no_fano_with_amp(xvec, *initial)))
            plt.plot(xvec,np.angle(S11exp_no_fano_with_amp(xvec, *initial)))
            initial2 = [0.0,kin,kex,0.0,tau,amp]
            plt.plot(xvec,np.angle(S11exp_no_fano_with_amp(xvec, *initial)))
            plt.show()

        initial = [0.0,kex,kin,phi,tau,amp]

    if type(ini) is not bool:    
        initial2 = np.array(ini)
        initial2[3] += 2.*np.pi*ini[4]*ini[0] #/delta
        initial2[0] = 0

        # print(initial2)
    # print(initial)
    popt,pcov,infodict,mesg,ier = fit.leastsq(residuals, initial, args=(xvec,yvec,y2vec),full_output=True)
    # print(popt)
    
    if 0:
        plt.figure()
        plt.plot(S11exp_no_fano_with_amp(xvec, *initial).real, S11exp_no_fano_with_amp(xvec, *initial).imag)
        plt.plot(S11exp_no_fano_with_amp(xvec, *popt).real, S11exp_no_fano_with_amp(xvec, *popt).imag)
        re, im = reim(yvec,y2vec)
        z = (re+1j*im)
        plt.plot(np.real(z), np.imag(z))
        plt.show()
    # print(pcov.shape)
    # print(pcov)

    ###
    xvec += f0 
    #xvec = xvec*delta + f0
    initial[0] = initial[0] + f0
    #initial[1] = initial[1]*delta
    #initial[2] = initial[2]*delta
    initial[3] = initial[3] - 2.*np.pi*initial[4]*f0 #/delta
    #initial[4] = initial[4]/delta
    popt[0] = popt[0] + f0
    #popt[1] = popt[1]*delta
    #popt[2] = popt[2]*delta
    popt[3] = popt[3] - 2.*np.pi*popt[4]*f0 #/delta
    #popt[4] = popt[4]/delta
    ###
    var = np.sum((np.absolute(yvec*np.exp(1j*y2vec) - S11exp_no_fano_with_amp(xvec, *popt)))**2)/(xvec.size-popt.size)
    if type(pcov)!=np.ndarray:
        # Experimental Andrea and Xuxin feature:
        print('WARNING: Could not estimate covariance matrix. Fitting without Fano')
        return S11exp_no_fano_fit(xvec,yvec,y2vec,ini=False)
        # pcov = np.zeros((np.size(popt),np.size(popt)))    
    perr = np.sqrt(np.diag(pcov)*var)
    #perr[0] = perr[0]*delta 
    #perr[1] = perr[1]*delta
    #perr[2] = perr[2]*delta
    #perr[3] = np.sqrt(perr[3]**2+(2.*np.pi*popt[4]*popt[5]/delta)**2)
    #perr[4] = perr[4]/delta
    ###
    zfit = S11exp_no_fano_with_amp(xvec, *popt)
    zini = S11exp_no_fano_with_amp(xvec, *initial)
    return popt,perr,zfit,zini

def circle_fit(data, **kwargs):
    """
    Analytical fit of a circle to the scattering data. Cf. Sebastian
    Probst: "Efficient and robust analysis of complex scattering data under
    noise in microwave resonators" (arXiv:1410.3365v2)

    :param data: The data.

    :return: A Parameters object containing the results.
    """
    # to be able to pass the data as an array of complex numbers
    zi = deepcopy(data)
    if len(np.shape(data)) != 1:
        zi = data[:,0] + 1j * data[:,1]

    # Normalize circle to deal with comparable numbers
    amp_norm = np.max(np.abs(zi))
    zi /= amp_norm

    # Calculate matrix of moments
    xi = zi.real
    yi = zi.imag
    xi_sqr = xi**2
    yi_sqr = yi**2
    zi = xi_sqr + yi_sqr
    Nd = float(len(xi))
    xi_sum = xi.sum()
    yi_sum = yi.sum()
    zi_sum = zi.sum()
    xiyi_sum = (xi*yi).sum()
    xizi_sum = (xi*zi).sum()
    yizi_sum = (yi*zi).sum()
    M =  np.array([
        [(zi*zi).sum(), xizi_sum, yizi_sum, zi_sum],
        [xizi_sum, xi_sqr.sum(), xiyi_sum, xi_sum],
        [yizi_sum, xiyi_sum, yi_sqr.sum(), yi_sum],
        [zi_sum, xi_sum, yi_sum, Nd]
    ])

    a0 = ((M[2][0]*M[3][2]-M[2][2]*M[3][0])*M[1][1]-M[1][2]*M[2][0]*M[3][1]-M[1][0]*M[2][1]*M[3][2]+M[1][0]*M[2][2]*M[3][1]+M[1][2]*M[2][1]*M[3][0])*M[0][3]+(M[0][2]*M[2][3]*M[3][0]-M[0][2]*M[2][0]*M[3][3]+M[0][0]*M[2][2]*M[3][3]-M[0][0]*M[2][3]*M[3][2])*M[1][1]+(M[0][1]*M[1][3]*M[3][0]-M[0][1]*M[1][0]*M[3][3]-M[0][0]*M[1][3]*M[3][1])*M[2][2]+(-M[0][1]*M[1][2]*M[2][3]-M[0][2]*M[1][3]*M[2][1])*M[3][0]+((M[2][3]*M[3][1]-M[2][1]*M[3][3])*M[1][2]+M[2][1]*M[3][2]*M[1][3])*M[0][0]+(M[1][0]*M[2][3]*M[3][2]+M[2][0]*(M[1][2]*M[3][3]-M[1][3]*M[3][2]))*M[0][1]+((M[2][1]*M[3][3]-M[2][3]*M[3][1])*M[1][0]+M[1][3]*M[2][0]*M[3][1])*M[0][2]
    a1 = (((M[3][0]-2.*M[2][2])*M[1][1]-M[1][0]*M[3][1]+M[2][2]*M[3][0]+2.*M[1][2]*M[2][1]-M[2][0]*M[3][2])*M[0][3]+(2.*M[2][0]*M[3][2]-M[0][0]*M[3][3]-2.*M[2][2]*M[3][0]+2.*M[0][2]*M[2][3])*M[1][1]+(-M[0][0]*M[3][3]+2.*M[0][1]*M[1][3]+2.*M[1][0]*M[3][1])*M[2][2]+(-M[0][1]*M[1][3]+2.*M[1][2]*M[2][1]-M[0][2]*M[2][3])*M[3][0]+(M[1][3]*M[3][1]+M[2][3]*M[3][2])*M[0][0]+(M[1][0]*M[3][3]-2.*M[1][2]*M[2][3])*M[0][1]+(M[2][0]*M[3][3]-2.*M[1][3]*M[2][1])*M[0][2]-2.*M[1][2]*M[2][0]*M[3][1]-2.*M[1][0]*M[2][1]*M[3][2])
    a2 = ((2.*M[1][1]-M[3][0]+2.*M[2][2])*M[0][3]+(2.*M[3][0]-4.*M[2][2])*M[1][1]-2.*M[2][0]*M[3][2]+2.*M[2][2]*M[3][0]+M[0][0]*M[3][3]+4.*M[1][2]*M[2][1]-2.*M[0][1]*M[1][3]-2.*M[1][0]*M[3][1]-2.*M[0][2]*M[2][3])
    a3 = (-2.*M[3][0]+4.*M[1][1]+4.*M[2][2]-2.*M[0][3])
    a4 = -4.

    def char_pol(x):
        return a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4

    def d_char_pol(x):
        return a1 + 2*a2*x + 3*a3*x**2 + 4*a4*x**3

    x0 = fit.newton(char_pol, 0., fprime=d_char_pol)

    M[3][0] = M[3][0] + 2*x0
    M[0][3] = M[0][3] + 2*x0
    M[1][1] = M[1][1] - x0
    M[2][2] = M[2][2] - x0

    U,s,Vt = np.linalg.svd(M)
    A_vec = Vt[np.argmin(s),:]

    xc = -A_vec[1]/(2.*A_vec[0])
    yc = -A_vec[2]/(2.*A_vec[0])
    # The term *sqrt term corrects for the constraint, because it may be
    # altered due to numerical inaccuracies during calculation
    r0 = 1./(2.*np.abs(A_vec[0]))*np.sqrt(A_vec[1]*A_vec[1]+A_vec[2]*A_vec[2]-4.*A_vec[0]*A_vec[3])

    return r0*amp_norm, xc*amp_norm, yc*amp_norm

def fit_delay(f, z, max_iter=0, smoothing_sigma=5, nbr_sigma_trim=1):
    """
    Recovers the phase delay by repeatedly fitting a circle to the complex z data.

    :param f: The array_like frequencies.
    :param z: The array_like complex values of the S-parameters.
    :param max_iter: Maximum number of iteration of Least-Squares to find the delay. Default to 0 (until tolerance is reached).
    :param smoothing_sigma: Smoothing sigma used for the guess of the delay for the gaussian kernel. Default to 5.
    :param nbr_sigma_trim: Number of width removed from the resonance for the fit of the slope. Default to 3.

    :return: The fitted time delay, whose unit is the inverse of the frequency unit.
    """
    guessed_delay, _ = guess_delay(f, z, smoothing_sigma=smoothing_sigma, nbr_sigma_trim=nbr_sigma_trim)

    def residuals(delay, f, z):
        z_temp = z * np.exp(1j*(2 * np.pi * delay * f))
        r0, xc, yc = circle_fit(z_temp)
        err = np.sum(np.abs(np.abs(z_temp - complex(xc, yc))-r0)**2)
        return err

    p_final = fit.leastsq(residuals, guessed_delay, args=(f, z), maxfev=max_iter, ftol=1e-15, xtol=1e-15)

    delay = p_final[0][0]

    # fig, ax = plt.subplots(figsize=(6*2/2.54, 4*2/2.54))
    # z_temp1 = z * np.exp(1j*(2 * np.pi * delay * f))
    # r0, xc, yc = circle_fit(z_temp1)
    # ax.plot(r0*np.cos(np.linspace(-np.pi, np.pi, 100))+xc, r0*np.sin(np.linspace(-np.pi, np.pi, 100))+yc, 'g')
    # ax.plot(np.real(z_temp1), np.imag(z_temp1), 'g.')
    # print(residuals(delay, f, z))

    # z_temp1 = z * np.exp(-1j*(2 * np.pi * delay * f))
    # r0, xc, yc = circle_fit(z_temp1)
    # ax.plot(r0*np.cos(np.linspace(-np.pi, np.pi, 100))+xc, r0*np.sin(np.linspace(-np.pi, np.pi, 100))+yc, 'b')
    # ax.plot(np.real(z_temp1), np.imag(z_temp1), 'b.')
    # print(residuals(-delay, f, z))

    # ax.set_aspect('equal', adjustable='box')
    # plt.tight_layout()
    # print(delay)


    return delay

def guess_delay(f, z, smoothing_sigma, nbr_sigma_trim):
    """
    Guesses the phase delay by repeatedly fitting the slope of the phase.

    :param f: The array_like frequencies.
    :param z: The array_like complex values of the S-parameters.
    :param smoothing_sigma: Smoothing sigma used for the guess of the delay for the gaussian kernel.
    :param nbr_sigma_trim: Number of width removed from the resonance for the fit of the slope.

    :return: The guessed time delay, whose unit is the inverse of the frequency unit.
    """
    # smoothen the data for initial guess
    if smoothing_sigma > 0:
        smoothed_r = np.array(gaussian_filter1d(z.real, smoothing_sigma))
        smoothed_i = np.array(gaussian_filter1d(z.imag, smoothing_sigma))
    else:
        smoothed_r = z.real
        smoothed_i = z.imag

    smoothed_z = np.array([x + 1j * y for x, y in zip(smoothed_r, smoothed_i)])

    # make smoothed phase vector removing 2pi jumps
    smoothed_phase = np.unwrap(np.angle(smoothed_z))
    smoothed_phase_deriv = np.diff(smoothed_phase)

    # get resonant frequency from maximum of the derivative of the smoothed phase
    avg_phase_deriv = np.average(smoothed_phase_deriv)
    err_phase_deriv = (smoothed_phase_deriv - avg_phase_deriv) ** 2
    idx_res = np.argmax(err_phase_deriv)

    guess_f_res = f[idx_res]

    idx_min, idx_max = get_width_from_phase(idx_res, err_phase_deriv)
    width = idx_max - idx_min
    idx_min = max(idx_min-nbr_sigma_trim*width, 10)
    idx_max = min(idx_max+nbr_sigma_trim*width, len(smoothed_phase)-10)

    slope_min, _, _, _, _ = linregress(f[:idx_min], smoothed_phase[:idx_min])
    slope_max, _, _, _, _ = linregress(f[idx_max:], smoothed_phase[idx_max:])
    slope = (slope_min + slope_max) / 2.
    guess_delay = -slope / (2 * np.pi)

    # print(TWO_PI * guess_delay * (f[-1] - f[0]))
    if 2 * np.pi * guess_delay * (f[-1] - f[0]) < 0.2:
        # threshold at which the data is almost a circle, where the optimization can do well on its own
        guess_delay = 0.

    return guess_delay, guess_f_res

def get_width_from_phase(idx_res, func):
    """
    Finds indices for peak width around given maximum position.

    :param idx_res: Array index of resonance.
    :param func: Real array with resonance data.

    :return: A tuple containing the indices of the lower and upper estimated edges of the resonance peak.
    """
    max_func = func[idx_res]
    avg_func = np.average(func)
    idx_min, idx_max = 0, len(func) - 1

    for i in range(idx_res, len(func)):
        if (max_func - func[i]) > (max_func - avg_func):
            idx_max = i
            break
    for i in range(idx_res, -1, -1):
        if (max_func - func[i]) > (max_func - avg_func):
            idx_min = i
            break
    return (idx_min, idx_max)














def S11exp_no_fano_fit(xvec,yvec,y2vec,ini=False): ###
    def residuals(params, x, y,y2):
        # return np.absolute(y*np.exp(1j*y2) - S11exp(x, *params))
        return np.absolute(y*np.exp(1j*y2) - S11exp_no_fano(x, *params))
    delta = np.amax(xvec)-np.amin(xvec)
    if type(ini) is list:
        f0 = ini[0]
        xvec = xvec - f0
        xvec = xvec/delta
        initial = ini
        initial[0] = 0.0
    else:
        re,im = reim(yvec,y2vec)
        x_max,sig = search_signal(xvec,re,im,filt=False)
        popt0, perr0, yfit0, yini0 = Lorentzian_fit(xvec,sig)
        
        if 0:
            plt.figure()
            plt.plot(xvec,sig,'b.')
            plt.plot(xvec,yfit0,'r')
            plt.show()            
        
        f0 = popt0[0]
        # print(x0)
        xvec = xvec - f0
        xvec = xvec/delta
        amp = popt0[1]
        lw = popt0[2]/delta
        kex = np.absolute(lw*(amp/2.))
        kin = np.absolute(lw*(1-amp/2.))
        tau = get_offset(np.diff(smooth(y2vec)))/(xvec[1]-xvec[0])/2./np.pi
        #phi = -np.pi + np.average(y2vec) - 2.*np.pi*xvec[0]*tau

        initial = [0.0,kex,kin,0.0,tau]
        phi =  y2vec[0] - np.angle(S11exp_no_fano(xvec[0], *initial))
        if 0:
            plt.figure()
            plt.plot(xvec,y2vec,'b.')
            plt.plot(xvec,2.*np.pi*tau*xvec)
            plt.plot()
            plt.show()        
        initial = [0.0,kex,kin,phi,tau]            
    popt,pcov,infodict,mesg,ier = fit.leastsq(residuals, initial, args=(xvec,yvec,y2vec),full_output=True)

    # print(pcov.shape)
    # print(pcov)

    ###
    xvec = xvec*delta + f0
    initial[0] = initial[0]*delta + f0
    initial[1] = initial[1]*delta
    initial[2] = initial[2]*delta
    initial[3] = initial[3] - 2.*np.pi*initial[4]*f0/delta
    initial[4] = initial[4]/delta
    popt[0] = popt[0]*delta + f0
    popt[1] = popt[1]*delta
    popt[2] = popt[2]*delta
    popt[3] = popt[3] - 2.*np.pi*popt[4]*f0/delta
    popt[4] = popt[4]/delta
    ###
    var = np.sum((np.absolute(yvec*np.exp(1j*y2vec) - S11exp_no_fano(xvec, *popt)))**2)/(xvec.size-popt.size) 
    if type(pcov)!=np.ndarray:
        pcov = np.zeros((np.size(popt),np.size(popt)))    
    perr = np.sqrt(np.diag(pcov)*var)
    perr[0] = perr[0]*delta 
    perr[1] = perr[1]*delta
    perr[2] = perr[2]*delta
    perr[3] = perr[3]
    perr[4] = perr[4]/delta
    ###
    zfit = S11exp_no_fano(xvec, *popt)
    zini = S11exp_no_fano(xvec, *initial)
    return popt,perr,zfit,zini
      
def S11Qexp_fit(xvec,yvec,y2vec,ini=False): ###
    def residuals(params, x, y,y2):
        return np.absolute(y*np.exp(1j*y2) - S11Qexp(x, *params))
    delta = np.amax(xvec)-np.amin(xvec)
    if type(ini) is list:
        f0 = ini[0]
        xvec = xvec - f0
        xvec = xvec/delta
        initial = ini
        initial[0] = 0.0
    else:
        re,im = reim(yvec,y2vec)
        x_max,sig = search_signal(xvec,re,im,filt=False)
        popt0, perr0, yfit0, yini0 = Lorentzian_fit(xvec,sig)
        
        if 0:
            plt.figure()
            plt.plot(xvec,sig,'b.')
            plt.plot(xvec,yfit0,'r')
            plt.show()            
        
        f0 = popt0[0]
        # print(x0)
        xvec = xvec - f0
        xvec = xvec/delta
        amp = popt0[1]
        lw = popt0[2]/delta
        kex = np.absolute(lw*(amp/2.))
        kin = np.absolute(lw*(1-amp/2.))
        tau = get_offset(np.diff(smooth(y2vec)))/(xvec[1]-xvec[0])/2./np.pi
        #phi = -np.pi + np.average(y2vec) - 2.*np.pi*xvec[0]*tau
        x0 = 0
        y0 = 0
        initial = [0.0,kex,(kin+kex)/2.,0.0,tau,x0,y0]
        phi =  y2vec[0] - np.angle(S11Qexp(xvec[0], *initial))
        if 0:
            plt.figure()
            plt.plot(xvec,y2vec,'b.')
            plt.plot(xvec,2.*np.pi*tau*xvec)
            plt.plot()
            plt.show()        
        initial = [0.0,kex,(kin+kex)/2.,phi,tau,x0,y0]            
    popt,pcov,infodict,mesg,ier = fit.leastsq(residuals, initial, args=(xvec,yvec,y2vec),full_output=True)

    ###
    xvec = xvec*delta + f0
    initial[0] = initial[0]*delta + f0
    initial[1] = initial[1]*delta
    initial[2] = initial[2]*delta
    initial[3] = initial[3] - 2.*np.pi*initial[4]*f0/delta
    initial[4] = initial[4]/delta
    popt[0] = popt[0]*delta + f0
    popt[1] = popt[1]*delta
    popt[2] = popt[2]*delta
    popt[3] = popt[3] - 2.*np.pi*popt[4]*f0/delta
    popt[4] = popt[4]/delta
    ###
    var = np.sum((np.absolute(yvec*np.exp(1j*y2vec) - S11Qexp(xvec, *popt)))**2)/(xvec.size-popt.size) 
    if type(pcov)!=np.ndarray:
        pcov = np.zeros((np.size(popt),np.size(popt)))     
    perr = np.sqrt(np.diag(pcov)*var)
    perr[0] = perr[0]*delta 
    perr[1] = perr[1]*delta
    perr[2] = perr[2]*delta
    perr[3] = np.sqrt(perr[3]**2+(2.*np.pi*popt[4]*x0/delta)**2)
    perr[4] = perr[4]/delta
    ###
    zfit = S11Qexp(xvec, *popt)
    zini = S11Qexp(xvec, *initial)
    return popt,perr,zfit,zini


def S11ge(x,f0,k1,kin,phi,tau,x0,y0,shift,pe):
    if k1>0 and kin>0 and pe>=0 and pe <= 1.0:
        return (1-pe)*np.exp(1j*phi+1j*2.*np.pi*tau*x)*((k1-kin)/2-1j*(x-f0))/((k1+kin)/2+1j*(x-f0))+pe*np.exp(1j*phi+1j*2.*np.pi*tau*x)*((k1-kin)/2-1j*(x-(f0+shift)))/((k1+kin)/2+1j*(x-(f0+shift)))+x0+1j*y0
    else:
        return 10**8*np.ones(np.size(x))
            
def S11ge_fit(xvec,yvec,y2vec,initial):
    #160221 sk
    def residuals(params, x, y,y2):
        return np.absolute(y*np.exp(1j*y2) - S11ge(x, *params))    
    popt,pcov,infodict,mesg,ier = fit.leastsq(residuals, initial, args=(xvec,yvec,y2vec),full_output=True)
    zfit = S11ge(xvec, *popt)
    zini = S11ge(xvec, *initial)    
    if type(pcov)!=np.ndarray:
        pcov = np.zeros((np.size(popt),np.size(popt)))  
    var = np.sum((np.absolute(yvec*np.exp(1j*y2vec) - S11ge(xvec, *popt)))**2)/(xvec.size-popt.size)        
    perr = np.sqrt(np.diag(pcov)*var)    
    return popt,perr,zfit,zini


def S11ratio(x,f0,k1,kin,shift):
    if k1>=0 and kin>=0:
        return (((k1-kin)/2-1j*(x-f0))/((k1+kin)/2+1j*(x-f0)))/(((k1-kin)/2-1j*(x-(f0+shift)))/((k1+kin)/2+1j*(x-(f0+shift))))
    else:
        return 10**8*np.ones(np.size(x)) 
            
def S11ratio_fit(xvec,yvec,y2vec,ini):
    #160221 sk
    def residuals(params, x, y,y2):
        return np.absolute(y*np.exp(1j*y2) - S11ratio(x, *params))    
    popt, pcov = fit.leastsq(residuals, ini, args=(xvec,yvec,y2vec))
    return popt,pcov
    
### multi peak fitting ########################################################
def Lorentzians(x, params):
    #gamma = FWHM
    #params = [offset,f0,a,gamma,f0,a,gamma,***]   
    Lorentzian = lambda x, x0, a, gamma: a/(((x-x0)/(gamma/2))**2 + 1)
    params =list(params)    
    offset = params.pop(0)
    yvec = np.tile([offset], np.size(x))
    for i in range(int((np.size(params))/3.0)):
        x0 =params[3*i]
        a = params[3*i+1]
        gamma = params[3*i+2]        
        #if a<0:
            #print '--> Lorentz_loop: Over range! (amp error:'+str(a)+')'
        #    return 10**8            
        yvec=np.add(yvec,Lorentzian(x, x0, a, gamma))
    return yvec


def doubleS11(x,f0,kex0,kin0,f1,kex1,kin1):
    return S11(x,f0,kex0,kin0)/S11(x,f1,kex1,kin1)

def doubleS11_fit(xvec,yvec,y2vec,ini):
    #201015 sk
    def residuals(params, x, y,y2):
        return np.absolute(y*np.exp(1j*y2) - doubleS11(x, *params))    
    popt, pcov = fit.leastsq(residuals, ini, args=(xvec,yvec,y2vec))
    return popt,pcov
   

###############################################################################
def Lorentzians_fit(xvec,yvec,r=0.1):
    '''
    if graph == True,each fitting results are showed.
    ###
    popt,pcov=mp.multifitting(xvec,yvec,fwhm=0.001,,graph=True)
    ###
    '''
    def e(params,x,y):
        residual = y-Lorentzians(x, params)
        return residual
        
    initial = get_initial(xvec,yvec,r)
    # print(initial)        
    popt,pcov,infodict,mesg,ier  = fit.leastsq(e, initial, args=(xvec,yvec),full_output=True)
    # print(popt)
    yfit = Lorentzians(xvec,popt)
    yini = Lorentzians(xvec,initial)
    if 0:
        plt.figure()
        plt.plot(xvec,yvec,'b.')
        plt.plot(xvec, yfit, 'r-', linewidth=1)
        plt.plot(xvec, yini, 'g-', linewidth=0.5)
        plt.show()
    var = np.sum((e(popt,xvec,yvec))**2)/(xvec.size-popt.size) 
    if type(pcov)!=np.ndarray:
        pcov = np.zeros(np.size(popt))     
    perr = np.sqrt(np.diag(pcov)*var)            
    return popt,perr,yfit,yini



def Lorentzian_rough(xvec, x0, a, gamma):
    if gamma>0.0:
        return a/(((xvec-x0)/(gamma/2))**2 + 1)
    else:
        return 10**8*np.ones(np.size(xvec))
        
def Lorentzian_rough_fit(xvec,yvec): 
    max_idx = np.argmax(np.absolute(yvec))
    ymax = yvec[max_idx]
    x0 = xvec[max_idx]
    xvec = xvec - x0
    idx_hwhm = np.argmin(np.absolute(yvec - ymax/2.0))
    gamma = 2.0 * np.absolute(xvec[max_idx]-xvec[idx_hwhm])
    a = ymax
    initial=[0.0,a,gamma]
    popt, pcov = fit.curve_fit(Lorentzian_rough, xvec, yvec, p0=initial)
    popt[0] = popt[0] + x0
    if type(pcov)!=np.ndarray:
        pcov = np.zeros(np.size(popt)) 
    perr = np.sqrt(np.diag(pcov))  
    return popt, perr
    
def get_initial(xvec, yvec,r):
    offset = get_offset(yvec)
    yveco = yvec-offset
    yvecoi = yveco
    ymax = np.absolute(np.amax(yveco))
    initial = [offset]
    num=0
    while 1:
        popt,perr = Lorentzian_rough_fit(xvec,yvecoi)
        if 0:        
            plt.figure()
            plt.plot(xvec,yvecoi,'b.-')
            plt.plot(xvec,Lorentzian_rough(xvec,*popt),'r')
            plt.show()    
        if (np.absolute(popt[1]) > r*ymax) and (popt[2] > (xvec[1]-xvec[0])):
            initial += list(popt)
            yvecoi -= Lorentzian_rough(xvec,*popt)
            num+=1
        else:
            break
    print('Peak num =',num)
    return initial   

def params_plot(popt,graph=False):
    '''
    if graph == True,each parame results are showed.
    pameters plot for popt from multifitting
    ###    
    num_list, f0_list, amp_list, fwhm_list=mp.parameplot(popt,graph=True)
    ###
    '''
    num = (np.size(popt)-1)/3
    f0_list=[]
    amp_list=[]
    fwhm_list=[]    
    for i in np.arange(num):
        f0_list.append(popt[3*i+1])        
        amp_list.append(popt[3*i+2]) 
        fwhm_list.append(popt[3*i+3]) 
    f0_list= np.array(f0_list)
    amp_list= np.array(amp_list)
    fwhm_list= np.array(fwhm_list) 
    num_list=np.arange(num)+1
    if graph:
        plt.figure()
        plt.plot(num_list,f0_list,'o')
        plt.title('Resonant frequency')
        plt.xlim(0,num+1)
        plt.show()
        
        plt.figure()
        plt.plot(num_list,amp_list,'o')
        plt.title('Amplitude')
        plt.xlim(0,num+1)
        plt.show()
        
        plt.figure()
        plt.plot(num_list,fwhm_list,'o')
        plt.title('FWHM')
        plt.xlim(0,num+1)
        plt.show()
    return num_list, f0_list, amp_list, fwhm_list 
    
