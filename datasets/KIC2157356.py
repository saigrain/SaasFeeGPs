import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pf
from astropy.table import Table
from glob import glob

files = glob('kplr002157356*.fits')

time = []
flux = []
error = []
quarter = []
for fl in files:
    hdul = pf.open(fl)
    t = hdul[1].data['TIME']
    f = hdul[1].data['PDCSAP_FLUX']
    e = hdul[1].data['PDCSAP_FLUX_ERR']
    q = hdul[1].data['SAP_QUALITY']
    l = np.isfinite(t) * np.isfinite(f) * np.isfinite(e) * (q == 0)
    time.append(t[l])
    flux.append(f[l])
    error.append(e[l])
    quarter.append(np.zeros(l.sum()) + int(hdul[0].header['QUARTER']))
    plt.plot(t[l], f[l], '.')

time = np.concatenate(time).ravel()
flux = np.concatenate(flux).ravel()
error = np.concatenate(error).ravel()
quarter = np.concatenate(quarter).ravel()

t = Table([time, flux, error, quarter], names=('time','flux','error','quarter'))
t.write('KIC2157356.txt',format='ascii')
plt.show()
