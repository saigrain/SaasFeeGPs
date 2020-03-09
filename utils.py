import numpy as np
import matplotlib.pyplot as pl
from scipy.spatial.distance import cdist
from numpy.linalg import inv
import george

def gauss1d(x,mu,sig):
    return np.exp(-(x-mu)**2/sig*2/2.)/np.sqrt(2*np.pi)/sig

def pltgauss1d(sig=1):
    mu=0
    x = np.r_[-4:4:101j]
    pl.figure(figsize=(10,7))
    pl.plot(x, gauss1d(x,mu,sig),'k-');
    pl.axvline(mu,c='k',ls='-');
    pl.axvline(mu+sig,c='k',ls='--');
    pl.axvline(mu-sig,c='k',ls='--');
    pl.axvline(mu+2*sig,c='k',ls=':');
    pl.axvline(mu-2*sig,c='k',ls=':');
    pl.xlim(x.min(),x.max());
    pl.ylim(0,1);
    pl.xlabel(r'$y$');
    pl.ylabel(r'$p(y)$');
    return

def gauss2d(x1,x2,mu1,mu2,sig1,sig2,rho):
    z = (x1-mu1)**2/sig1**2 + (x2-mu2)**2/sig2**2 - \
      2*rho*(x1-mu1)*(x2-mu2)/sig1/sig2    
    e = np.exp(-z/2/(1-rho**2))
    return e/(2*np.pi*sig1*sig2*np.sqrt(1-rho**2))

def pltgauss2d(rho=0,show_cond=0):
    mu1, sig1 = 0,1
    mu2, sig2 = 0,1
    y2o = -1
    x1 = np.r_[-4:4:101j]
    x2 = np.r_[-4:4:101j]
    x22d,x12d = np.mgrid[-4:4:101j,-4:4:101j]
    y = gauss2d(x12d,x22d,mu1,mu2,sig1,sig2,rho)
    y1 = gauss1d(x1,mu1,sig1)
    y2 = gauss1d(x2,mu2,sig2)
    mu12 = mu1+rho*(y2o-mu2)/sig2**2
    sig12 = np.sqrt(sig1**2-rho**2*sig2**2)
    y12 = gauss1d(x1,mu12,sig12)
    pl.figure(figsize=(10,10))
    ax1 = pl.subplot2grid((3,3),(1,0),colspan=2,rowspan=2,aspect='equal')
    v = np.array([0.02,0.1,0.3,0.6]) * y.max()
    CS = pl.contour(x1,x2,y,v,colors='k')
    if show_cond: pl.axhline(y2o,c='r')
    pl.xlabel(r'$y_1$');
    pl.ylabel(r'$y_2$');
    pl.xlim(x1.min(),x1.max())
    ax1.xaxis.set_major_locator(pl.MaxNLocator(5, prune = 'both'))
    ax1.yaxis.set_major_locator(pl.MaxNLocator(5, prune = 'both'))
    ax2 = pl.subplot2grid((3,3),(0,0),colspan=2,sharex=ax1)
    pl.plot(x1,y1,'k-')
    if show_cond: pl.plot(x1,y12,'r-')
    pl.ylim(0,0.8)
    pl.ylabel(r'$p(y_1)$')
    pl.setp(ax2.get_xticklabels(), visible=False)
    ax2.xaxis.set_major_locator(pl.MaxNLocator(5, prune = 'both'))
    ax2.yaxis.set_major_locator(pl.MaxNLocator(4, prune = 'upper'))
    pl.xlim(x1.min(),x1.max())
    ax3 = pl.subplot2grid((3,3),(1,2),rowspan=2,sharey=ax1)
    pl.plot(y2,x2,'k-')
    if show_cond: pl.axhline(y2o,c='r')
    pl.ylim(x2.min(),x2.max());
    pl.xlim(0,0.8);
    pl.xlabel(r'$p(y_2)$')
    pl.setp(ax3.get_yticklabels(), visible=False)
    ax3.xaxis.set_major_locator(pl.MaxNLocator(4, prune = 'upper'))
    ax3.yaxis.set_major_locator(pl.MaxNLocator(5, prune = 'both'))
    pl.subplots_adjust(hspace=0,wspace=0)
    return 

def SEKernel(par, x1, x2):
    A, Gamma = par
    D2 = cdist(x1.reshape(len(x1),1), x2.reshape(len(x2),1), 
               metric = 'sqeuclidean')
    return A * np.exp(-Gamma*D2)

def Pred_GP(CovFunc, CovPar, xobs, yobs, eobs, xtest):
    # evaluate the covariance matrix for pairs of observed inputs
    K = CovFunc(CovPar, xobs, xobs) 
    # add white noise
    K += np.identity(xobs.shape[0]) * eobs**2
    # evaluate the covariance matrix for pairs of test inputs
    Kss = CovFunc(CovPar, xtest, xtest)
    # evaluate the cross-term
    Ks = CovFunc(CovPar, xtest, xobs)
    # invert K
    Ki = inv(K)
    # evaluate the predictive mean
    m = np.dot(Ks, np.dot(Ki, yobs))
    # evaluate the covariance
    cov = Kss - np.dot(Ks, np.dot(Ki, Ks.T))
    return m, cov

def Plot_Pred_GP(CovFunc, CovPar, xobs, yobs, eobs, xtest):
    pl.errorbar(xobs,yobs,yerr=eobs,capsize=0,fmt='k.',label='observed')
    if len(xobs) > 0:
        m, C = Pred_GP(CovFunc,CovPar,xobs,yobs,eobs,xtest)
    else:
        m = np.zeros(len(xtest))
        C = CovFunc(CovPar,xtest,xtest)
    sig = np.sqrt(np.diag(C))
    if len(xtest) < 10:
        pl.errorbar(xtest, m, yerr=sig,fmt='.',capsize=0,color='C0',label='predicted')
    else:
        pl.plot(xtest, m,'C0-',label='predictive mean')
        pl.fill_between(xtest,m+sig,m-sig,color='C0',alpha=0.2,label='1-$\sigma$ confidence interval')
    pl.xlabel(r'$x$')
    pl.ylabel(r'$y$')
    pl.legend(loc=0)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def kernel_SE(X1,X2,par):
    p0 = 10.0**par[0]
    p1 = 10.0**par[1]
    D2 = cdist(X1,X2,'sqeuclidean')
    K = p0 * np.exp(- p1 * D2)
    return np.matrix(K)
def kernel_Mat32(X1,X2,par):
    p0 = 10.0**par[0]
    p1 = 10.0**par[1]
    DD = cdist(X1, X2, 'euclidean')
    arg = np.sqrt(3) * abs(DD) / p1
    K = p0 * (1 + arg) * np.exp(- arg)
    return np.matrix(K)
def kernel_RQ(X1,X2,par):
    p0 = 10.0**par[0]
    p1 = 10.0**par[1]
    alpha = 10.00**par[2]
    D2 = cdist(X1, X2, 'sqeuclidean')
    K = p0 * (1 + D2 / (2*alpha*p1))**(-alpha)
    return np.matrix(K)
def kernel_Per(X1,X2,par):
    p0 = 10.0**par[0]
    p1 = 10.0**par[1]
    period = par[2]
    DD = cdist(X1, X2, 'euclidean')
    K = p0 * np.exp(- p1*(np.sin(np.pi * DD / period))**2) 
    return np.matrix(K)
def kernel_QP(X1,X2,par):
    p0 = 10.0**par[0]
    p1 = 10.0**par[1]
    period = par[2]
    p3 = 10.0**par[3]
    DD = cdist(X1, X2, 'euclidean')
    D2 = cdist(X1, X2, 'sqeuclidean')
    K = p0 * np.exp(- p1*(np.sin(np.pi * DD / period))**2 - p3 * D2)
    return np.matrix(K)
def add_wn(K,lsig):
    sigma=10.0**lsig
    N = K.shape[0]
    return K + sigma**2 * np.identity(N)
def get_kernel(name):
    if name == 'SE': return kernel_SE
    elif name == 'RQ': return kernel_RQ
    elif name == 'M32': return kernel_Mat32
    elif name == 'Per': return kernel_Per
    elif name == 'QP': return kernel_QP
    else: 
        print('No kernel called {:s} - using SE'.format(name))
        return kernel_SE

def pltsamples(kernel = 'SE', par1 = 0.0, par2 = 0.0, wn = -3, par3 = 0.0,par4 = 0.0):
    x = np.r_[-5:5:201j]
    X = np.matrix([x]).T # scipy.spatial.distance expects matrices
    kernel_func = get_kernel(kernel)
    par = np.array([par1,par2,par3,par4])
    K = kernel_func(X,X,par)
    K = add_wn(K,wn)
    fig = pl.figure(figsize=(12,4))
    ax1 = pl.subplot2grid((1,3), (0, 0))
    pl.plot(x,K[:,100])
    pl.xlabel("$x-x'$")
    pl.ylabel("$k(x,x')$")
    pl.title('Covariance function')
    ax2 = pl.subplot2grid((1,3), (0, 1), aspect='equal')
    pl.imshow(K)
    pl.title('Covariance matrix')
    pl.colorbar()
    ax3 = pl.subplot2grid((1,3), (0,2))
    np.random.seed(0)
    for i in range(3):
        y = np.random.multivariate_normal(np.zeros(len(x)),K)
        y -= y.mean()
        pl.plot(x,y-3*np.sqrt(10.0**par[0])*i)
    pl.xlim(-5,5)
    # pl.ylim(-8*np.sqrt(10.0**par[0]),5*np.sqrt(10.0**par[0]))
    pl.xlabel('x')
    pl.ylabel('y')
    pl.title('Samples from %s prior' % kernel)
    pl.tight_layout()
