'''
Created on Jun 20, 2014

@author: tel
'''
import os,re
import numpy as np
import morphsnakes as ms
from matplotlib import pyplot as ppl
from PIL import Image
from scipy import ndimage, interpolate
from skimage import filter
from skimage.morphology import skeletonize

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
        img = Image.open(fpath)
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
        #arr_w_frame = np.zeros(np.array(input_arr.shape)+20) + 8000
        #arr_w_frame[10:-10,10:-10] = input_arr
        obj = np.asarray(input_arr).view(subtype)
        #obj = np.ndarray.__new__(subtype,shape,dtype,buffer,offset,strides,order)
        return obj
    
    def __array_finalize__(self,obj):
        if obj is None: 
            return
        #if we wanted to add an attribute to our subclass we would need a line like the following here
        #self.info = getattr(obj, 'info', None)
    
    def AddCountour1(self):
        num_iters = 100
#         val = filter.threshold_otsu(self)
#         mask = self < val
#          
#         open_mask = ndimage.binary_opening(mask)
#         close_mask = ndimage.binary_closing(open_mask)
        gauss = ndimage.gaussian_filter(self, 2)
        gauss = gauss.astype(float)
        gauss = gauss - gauss.min()
        gauss = gauss/gauss.max()
#         for i in np.nditer(gauss, op_flags=['readwrite']):
#             if i < .6:
#                 i[...] = 0

        # g(I)
        gI = ms.gborders(gauss, alpha=1000, sigma=5)
        
        # Morphological GAC. Initialization of the level-set.
        mgac = ms.MorphGAC(gI, smoothing=2, threshold=.244, balloon=-1)
        mgac.levelset = ms.circle_levelset(self.shape, np.array(self.shape)/2, max(self.shape)/2.66666, scalerow=.75)
        
        # non-visual evolution
#         for i in range(num_iters):
#             mgac.step()
#         print 'hey'
        
        # Visual evolution.
        ppl.figure()
        ms.evolve_visual(mgac, num_iters=num_iters, background=self)
        self.msnake = ms
        
    def AddCountour2(self):
        val = filter.threshold_otsu(self)
        mask = self < val
        
        # Morphological ACWE. Initialization of the level-set.
        macwe = ms.MorphACWE(self, smoothing=1, lambda1=1, lambda2=1)
        macwe.levelset = ms.circle_levelset(self.shape, (20, 20), 25)
        
        for i in range(20):
            macwe.step()
        print 'hey'
        
        # Visual evolution.
#         ppl.figure()
#         ms.evolve_visual(macwe, num_iters=190, background=self)
    def AddCenterline(self):
        x,y = [],[]
        it = np.nditer(skeletonize(self.msnake.levelset), flags=['multi_index'])
        while not it.finished:
            if it[0]:
                x.append(it.multi_index[1])
                y.append(it.multi_index[0])
            it.iternext()
        x.reverse()
        y.reverse()
        zipped = zip(x,y)
        zipped.sort()
        centerdict = {}
        for xy in zipped:
            if not xy[0] in centerdict:
                centerdict[xy[0]] = xy[1]
        centerline = centerdict.items()
        centerline.sort()
        tck = interpolate.splrep(zip(*centerline[len(centerline)/4:-len(centerline)/4])[0], zip(*centerline[len(centerline)/4:-len(centerline)/4])[1],k=1,s=100)
        xnew = np.arange(0, self.shape[1], self.shape[1]/200.0)
        ynew = interpolate.splev(xnew, tck, der=0)
        self.centerline = zip(xnew,ynew)
    
    #def AddContour(self):
        #ax1.contour(msnake.levelset, [0.5], colors='r')
    
if __name__=='__main__':
    cell = Cell('exampledata/cell5')
    cell.bf[1][0].AddCountour1()
    ppl.show()