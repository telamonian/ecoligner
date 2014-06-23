'''
Created on Jun 20, 2014

@author: tel
'''
import os,re
import numpy as np
import morphsnakes as ms
from PIL import Image
from scipy import ndimage
from skimage import filter
from matplotlib import pyplot as ppl

np.set_printoptions(suppress=True,linewidth=10**6,threshold=10**6)

class Cell(object):
    def __init__(self,dirpath):
        #this loop produces three dictionaries, self.bf, self.conc, self.rfp, that contain the relevant image data in lists of augmented numpy arrays
        walk = os.walk(dirpath).next()
        for fname in walk[2]:
            typere = re.search('((?:rfp)|(?:conc)|(?:bf))-(\d+)\.tif', fname)
            if typere:
                tmp_dict = getattr(self, typere.group(1), {})
                tmp_dict[int(typere.group(2))] = Frames(os.path.join(walk[0], fname))
                setattr(self, typere.group(1), tmp_dict)
                
class Frames(object):
    def __init__(self,fpath):
        self._frames = []
        img=Image.open(fpath)
        i = 0
        while(True):
            try:
                img.seek(i)
                self.append(Frame(img))
                i+=1
            except EOFError:
                break
            
    def __getitem__(self,i):
        return self._frames[i]
        
    def __setitem__(self,i,val):
        self._frames[i] = val
        
    def append(self,val):
        self._frames.append(val)

class Frame(np.ndarray):
    def __new__(subtype,img,dtype=float,buffer=None,offset=0,strides=None,order=None):
        input_arr = np.array(img.getdata()).reshape(img.size[::-1])
        obj = np.asarray(input_arr).view(subtype)
        #shape = img.size[::-1]
        #obj = np.ndarray.__new__(subtype,shape,dtype,buffer,offset,strides,order)
        return obj
    
    def __array_finalize__(self,obj):
        if obj is None: 
            return
        #if we wanted to add an attribute to our subclass we would need a line like the following here
        #self.info = getattr(obj, 'info', None)
    
    def AddCountour1(self):
#         val = filter.threshold_otsu(self)
#         mask = self < val
#          
#         open_mask = ndimage.binary_opening(mask)
#         close_mask = ndimage.binary_closing(open_mask)
        gauss = ndimage.gaussian_filter(self, 2)
        gauss = gauss.astype(float)
        gauss = gauss - gauss.min()
        gauss = gauss/gauss.max()
        # g(I)
        gI = ms.gborders(gauss, alpha=1000, sigma=5)
        
        # Morphological GAC. Initialization of the level-set.
        mgac = ms.MorphGAC(gI, smoothing=2, threshold=.24, balloon=-1)
        mgac.levelset = ms.circle_levelset(self.shape, (20, 20), 15, scalerow=.75)
        
        # Visual evolution.
        ppl.figure()
        ms.evolve_visual(mgac, num_iters=110, background=self)
    
    def AddCountour2(self):
        val = filter.threshold_otsu(self)
        mask = self < val
        
        # Morphological ACWE. Initialization of the level-set.
        macwe = ms.MorphACWE(self, smoothing=1, lambda1=1, lambda2=1)
        macwe.levelset = ms.circle_levelset(self.shape, (20, 20), 25)
        
        # Visual evolution.
        ppl.figure()
        ms.evolve_visual(macwe, num_iters=190, background=self)
        
if __name__=='__main__':
    cell = Cell('exampledata/cell12')
    cell.bf[1][0].AddCountour1()
    ppl.show()