'''
Created on Jun 20, 2014

@author: tel
'''
import os,re
import numpy as np
import morphsnakes as ms
import shape
from matplotlib import pyplot as ppl
from PIL import Image
from scipy import ndimage, interpolate
from skimage import filter
from skimage.morphology import skeletonize

np.set_printoptions(suppress=True,linewidth=10**6,threshold=10**6)

class Cell(object):
    '''
    A cell object contains a number of Frames attributes named according to the scheme used to name the image files (cell.bf, cell.rfp, etc.)
    Example usage:
        cell = Cell('exampledata/cell5')    # dirpath is path to image data
        cell.bf[1][0].AddContour()          # first index is based on image file name, second index is on a zero-indexed list of Frame objects created from individual frames in image file
    '''
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
        input_arr = np.array(img.getdata()).reshape(img.size[::-1])[::-1]
        #arr_w_frame = np.zeros(np.array(input_arr.shape)+20) + np.mean(input_arr)
        #arr_w_frame[10:-10,10:-10] = input_arr
        obj = np.asarray(input_arr).view(subtype)
        #obj = np.ndarray.__new__(subtype,shape,dtype,buffer,offset,strides,order)
        return obj
    
    def __array_finalize__(self,obj):
        if obj is None: 
            return
        #if we wanted to add an attribute to our subclass we would need a line like the following here
        #self.info = getattr(obj, 'info', None)
        
    def AddContour(self, filtered=True, doMGAC=True, doMACWE=False, maxIters=100):
        if filtered:
            self.Filtered()
        if doMGAC:
            self.DoMGAC(maxIters)
        if doMACWE:
            self.DoMACWE(maxIters)

    def Filtered(self):
        gauss = ndimage.gaussian_filter(self, 2)
        gauss = gauss.astype(float)
        gauss = gauss - gauss.min()
        self.filtered = gauss/gauss.max()
            
    def DoMGAC(self, maxIters):
        gI = ms.gborders(self.filtered, alpha=1000, sigma=5)
        
        # Morphological GAC. Initialization of the level-set.
        mgac = ms.MorphGAC(gI, smoothing=2, threshold=.244, balloon=-1)
        mgac.levelset = ms.circle_levelset(self.shape, np.array(self.shape)/2, max(self.shape)/2.66666, scalerow=.75)
        
        # non-visual evolution
        converged = False
        convergeFrames = 10
        levelsets = np.zeros(self.shape + (convergeFrames,))
        for i in range(convergeFrames - 1):
            mgac.step()
            levelsets[:,:,i] = mgac.levelset
        levelsets = np.roll(levelsets, 1, axis=2)
        for i in range(maxIters - convergeFrames):
            mgac.step()
            levelsets[:,:,i%convergeFrames] = mgac.levelset
            diffs = []
            for j in range(convergeFrames - 1):
                diff = np.max(np.sum(np.abs(levelsets - np.roll(levelsets, j+1, axis=2)), axis=(0,1)))
                if diff < mgac.levelset.size/100:
                    diffs.append(diff)
                else:
                    break
            if len(diffs)==(convergeFrames - 1):
                converged = True
                break
            
        self.mgac = mgac
        self.levelset = self.mgac.levelset
        self.contour = shape.ContourShape(self.levelset)
        return converged
    
    def DoMACWE(self, maxIters):
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
        self.centerLine = shape.CenterShape(self.levelset)
    
    def AddPoles(self):
        self.poles = shape.PoleShape(self.contour.GetIntersect(self.centerLine))
    
    def AddRibs(self):
        self.ribs = shape.RibsShape(self.levelset, self.centerLine, self.poles)
    
    def AddCapsule(self):
        pass

    def AddContour1(self):
        num_iters = 5
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
        for i in range(num_iters):
            mgac.step()
            print i
            print mgac.levelset
        print 'hey'
        
        # Visual evolution.
#         ppl.figure()
#         ms.evolve_visual(mgac, num_iters=num_iters, background=self)
#         self.msnake = ms
        
    def AddContour2(self):
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
    
    def Segment(self):
        self.AddContour()
        self.AddCenterline()
        self.AddPoles()
        
    def Plot(self):
        shape.SetBackground(self)
        self.contour.PlotSegs()
        self.centerLine.PlotSegs()
        self.poles.PlotPoints()
        shape.ShowPlot()
    
if __name__=='__main__':
    cell = Cell('exampledata/cell5')
    cell.bf[1][0].Segment()
    cell.bf[1][0].Plot()
#     cell.bf[1][0].AddContour()
#     cell.bf[1][0].AddCenterline()
#     cell.bf[1][0].AddPoles()
#     cell.bf[1][0].AddRibs()
#     print cell.bf[1][0].levelset
#     cell.bf[1][0].contour.PlotSegs()
#     cell.bf[1][0].centerLine.PlotSegs()
#     shape.Shape.ax0.set_xlim((0,40))
#     shape.Shape.ax0.set_ylim((0,40))
#     shape._plt.show()
#     cell.bf[1][0].contour.PlotGrid((40,40), background=cell.bf[1][0])
#     cell.bf[1][0].centerLine.PlotGrid((40,40))
#     cell.bf[1][0].poles.PlotGrid((40,40))
#     for rib in cell.bf[1][0].ribs.ribList:
#         rib.PlotGrid((40,40))
#     shape.Shape.plt.show()
#     print cell.bf[1][0].contour.GetGrid((40,40))
#     cell.bf[1][0].contour.PlotGrid((40,40), background=cell.bf[1][0])
#     print cell.bf[1][0].centerLine.GetGrid((40,40))
#     cell.bf[1][0].centerLine.PlotGrid((40,40), background=cell.bf[1][0])
#     print cell.bf[1][0].pole.GetGrid((40,40))
    #cell.bf[1][0].Addcontour1()
    #ppl.show()