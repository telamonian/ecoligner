'''
Created on Jun 20, 2014

@author: tel
'''
from copy import deepcopy
from PIL import Image
import numpy as np
import os
import re
from scipy import ndimage
import scipy.io as sio
from skimage import filter

from src import morphsnakes as ms
from src import shape

threshold = .244
np.set_printoptions(suppress=True,linewidth=10**6,threshold=10**6)

class Cell(object):
    '''
    A cell object contains a number of Frames attributes named according to the scheme used to name the image files (cell.bf, cell.rfp, etc.)
    Example usage:
        cell = Cell('exampledata/cell5')    # dirpath is path to image and trajectory data
        cell.bf[1][0].AddContour()          # first index is based on image file name, second index is on a zero-indexed list of Frame objects created from individual frames in image file
    '''
    def __init__(self,dirpath):
        #this loop produces three dictionaries, self.bf, self.conc, self.rfp, that contain the relevant image data in lists of augmented numpy arrays
        walk = os.walk(dirpath).next()
        for fname in walk[2]:
            imgTypeRe = re.search('((?:rfp)|(?:conc)|(?:bf))-(\d+)\.tif', fname)
            traceRe = re.search('(\S*)(\d+)(\S*).mat', fname)
            if imgTypeRe:
                tmp_dict = getattr(self, imgTypeRe.group(1), {})
                tmp_dict[int(imgTypeRe.group(2))] = Frames(os.path.join(walk[0], fname))
                setattr(self, imgTypeRe.group(1), tmp_dict)
            if traceRe:
                if '_transformed' not in traceRe.group(0):
                    tmp_dict = getattr(self, 'trajectories', {})
                    tmp_dict[int(traceRe.group(2))] = Trajectories(os.path.join(walk[0], fname))
                    setattr(self, 'trajectories', tmp_dict)
                
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
        obj = np.asarray(input_arr).view(subtype)
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
#         mgac = ms.MorphGAC(gI, smoothing=2, threshold=.244, balloon=-1)
        mgac = ms.MorphGAC(gI, smoothing=2, threshold=threshold, balloon=-1)
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
        print self.levelset
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
        self.poles = shape.PoleShape(self.contour, self.centerLine)
    
    def AddRibs(self):
        self.ribs = shape.RibsShape(self.contour, self.centerLine, self.poles)
    
    def AddCapsule(self):
        self.capsule = shape.CapsuleShape(self.poles, self.ribs)
        
    def AddCapsuleRibs(self):
        self.capsuleRibs = shape.RibsShape(self.capsule, self.centerLine, self.poles)

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
        self.AddRibs()
        self.AddCapsule()
        self.AddCapsuleRibs()
        
    def Plot(self):
        print 'threshold: %.3f' % threshold
        print 'quality: %.3f' % self.ribs.Quality(self.capsuleRibs)
        print 'length/width ratio: %.3f' % (float(self.poles.lines[0].length)/self.capsule.radius)
        self.contour.PlotSegs(c='r',lw=2)
        self.centerLine.PlotSegs()
        self.poles.PlotPoints()
        self.ribs.PlotSegs(c='y')
        self.capsule.PlotConnectedPoints(c='g',lw=12,alpha=.75)
        shape.SetBackground(self)

class Trajectories(object):
    def __init__(self,fpath):
        self.fpath = fpath
        trajData = sio.loadmat(self.fpath)
        self._trajectories = [Trajectory(coord[0][:,1:3], reverseY=40) for coord in trajData.itervalues().next()[0,0]['TracksROI']['Coordinates']]
                
    def __getitem__(self,i):
        return self._trajectories[i]
        
    def __setitem__(self,i,val):
        self._trajectories[i] = val
        
    def append(self,val):
        self._trajectories.append(val)
    
    def Plot(self, frame):
        self.PlotSegs()
        self.PlotSegsTrans()
        # add a transformed capsule shape for easy comparison
        transCap = deepcopy(frame.capsule)
        transCap.OverlayOnCell(frame.capsule)
        transCap.PlotConnectedPoints(c='g',lw=12,alpha=.75)
    
    def PlotSegs(self, attr='trajShape'):
        shape.Shape.ResetColor()
        for traj in self._trajectories:
            traj.__getattribute__(attr).PlotSegs(c='cycle')
    
    def PlotSegsTrans(self):
        self.PlotSegs('transTrajShape')
    
    def SaveTransform(self):
        outfile = ''.join(self.fpath.split('.')[:-1]) + '_transformed.mat'
        trajData = sio.loadmat(self.fpath)
        for coord,traj in zip(trajData.itervalues().next()[0,0]['TracksROI']['Coordinates'], self._trajectories):
            coord[0][:,1:3] = traj.transTrajShape.points
        sio.savemat(outfile, trajData)
    
    def Transform(self, frame):
        for traj in self._trajectories:
            traj.transTrajShape.OverlayOnCell(frame.capsule)
        
class Trajectory(np.ndarray):
    def __new__(subtype,path,reverseY=None,dtype=float,buffer=None,offset=0,strides=None,order=None):
        '''reverseY deals with the situation where the origin of the path is in the upper left corner of the image rather than the lower left. set it to the height of the image'''
        input_arr = np.array(path)
        if reverseY:
            input_arr[:,1] = reverseY - input_arr[:,1]
        obj = np.asarray(input_arr).view(subtype)
        return obj
    
    def __array_finalize__(self,obj):
        if obj is None: 
            return
        self.trajShape = getattr(obj, 'trajShape', shape.TrajectoryShape(obj.real))
        self.transTrajShape = getattr(obj, 'transTrajShape', shape.TrajectoryShape(obj.real))

if __name__=='__main__':
    threshold = .23 # threshold is the primary parameter that determines how well the segmentation works. A value somewhere between .2 and .3 seems to be good for most cells
    while True:
        try:
            cellDataDir = 'exampledata/cell21' # path to directory containing cell data. you can batch process multiple cells by using a path to a directory that contains data directories (i.e. 'exampledata'), but probably shouldn't at this point in the program's evolution
            
            cell = Cell(cellDataDir)
            cell.bf[1][0].Segment()
            cell.bf[1][0].Plot()
            try:
                for trajList in cell.trajectories.itervalues():
                    trajList.Transform(cell.bf[1][0])
                    trajList.SaveTransform()
                cell.trajectories[1].Plot(cell.bf[1][0])
            except AttributeError:
                pass
            #shape.AutoscaleAxes()
            shape.SetBackground(cell.bf[1][0])
            shape.SavePlot(os.path.join(cellDataDir,'segmented.png'))
            #shape.ShowPlot()
            break
        except IndexError:
            threshold+=.01