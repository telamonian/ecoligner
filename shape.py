'''
Created on Jun 24, 2014

@author: tel
'''
import itertools
import math
from matplotlib import _cntr as cntr
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as __plt
import numpy as np
from scipy import interpolate
from skimage.morphology import skeletonize

_plt = __plt
_fig = _plt.figure(1)
_ax0 = _plt.subplot(gridspec.GridSpec(1,1,wspace=0.15,hspace=0.07)[0,0])

def SetBackground(background):
    _plt.imshow(background, cmap=_plt.cm.gray, origin='lower')
    
def ShowPlot():
    _plt.show()

class Shape(object):
#     __metaclass__ = SharedPltMetaclass
    
#     _plt = plt
#     _fig = plt.figure(1)
#     _ax0 = plt.subplot(gridspec.GridSpec(1,1,wspace=0.15,hspace=0.07)[0,0])
    
    close = False
    # segLen is the length of the walk that makes up one of the Shape's segs.
    # -1=inf (closed shape), 0=point, 1=line segment, 2=two joined segments, etc.
    segLen = -1
    
    def __init__(self, points):
        self.points = np.array(points)
        self.GenSegs()
        self.CalcDist()
    
    @property
    def plt(self):
        return _plt
    
    @property
    def fig(self):
        return _fig
    
    @property
    def ax0(self):
        return _ax0
            
    def CalcDist(self):
        if self.segLen==0:
            self.maxDist = 0
            self.avgDist = 0
            self.stdDist = 0
        else:
            dists = []
            for seg in self.segs:
                dists.append(seg.length)
#             for pa, pb in zip(self.points[:-1], self.points[1:]):
#                 dists.append(self.Dist(pa,pb))
            self.maxDist = np.max(dists)
            self.avgDist = np.mean(dists)
            self.stdDist = np.std(dists)
        
    def GenSegs(self):
        self.segs = []
        if self.segLen < 0:
            for pa,pb in zip(self.points[:-1], self.points[1:]):
                self.segs.append(Seg(pa, pb))
            if self.close:
                self.segs.append(Seg(self.points[-1], self.points[0]))
        elif self.segLen==0:
            pass
        else:
            for i in range(len(self.points))[::self.segLen+1]:
                for j in range(self.segLen):
                    self.segs.append(Seg(*self.points[i+j:i+j+2]))
        
    def GetClosest(self, other):
        '''return a list, length len(self.points), of pairs of closest points in self, other in format [[self index, other index], ...]'''
        ret = []
        if len(self.points)<1 or len(other.points)<1:
            return ret
        for i,sp in enumerate(self.points):
            mindex = 0
            mindist = self.__class__.Dist(sp, other.points[mindex])
            for j,op in enumerate(other.points[1:]):
                dist = self.__class__.Dist(sp, op)
                if dist < mindist:
                    mindex = j
                    mindist = dist
            ret.append([i, mindex])
        return ret
    
    def GetIntersect(self, other):
        ret = []
        for sSeg,oSeg in ((sSeg,oSeg) for sSeg in self.segs for oSeg in other.segs):
            intersect = sSeg.GetIntersect(oSeg)
            if type(intersect)!=type(False):
                ret.append(intersect)
        return ret
    
    def GetGrid(self, dims, transpose=False):
        flipBit = 1 - 2*transpose
        ret = np.zeros(dims[::flipBit])
        for seg in self.segs:
            try:
                for p in (np.floor(p) for p in seg):
                    if (0<=p[0]<dims[::flipBit][0]) and (0<=p[1]<=dims[::flipBit][1]):
                        ret.__setitem__(tuple(p[::flipBit].tolist()), 1)
            except IndexError:
                pass
#             ret = np.logical_or(ret, interpolate.griddata(seg + [seg[1] + np.array((1,0))] + [seg[0] + np.array((1,0))] , (True,)*4, tuple(np.mgrid[:shape[0],:shape[1]]), fill_value=False))
        return ret
    
    def GetLength(self, i1, i2):
        '''gives the length of the line segments between points with index i1 and i2'''
        ret = 0
        i1,i2 = sorted((i1,i2))
        for seg in self.segs[i1:i2]:
            ret+=seg.length
        return ret
    
    def PlotGrid(self, dims, background=None, transpose=False):
        flipBit = 1 - 2*transpose
        for seg in self.segs:
            self.plt.plot(*zip(*seg)[::flipBit], c='r')
#         plt.axis([0,dims[0],0,dims[1]])
        if background is not None:
            self.plt.imshow(background, cmap=_plt.cm.gray, origin='lower')
        self.plt.show()
    
    def PlotPoints(self, radius=.7, fc='g'):
        for point in self.points:
            circle = _plt.Circle(point, radius, fc=fc)
            self.ax0.add_patch(circle)
    
    def PlotSegs(self):
        for seg in self.segs:
            line = _plt.Line2D(*zip(*seg)[:2])
            self.ax0.add_line(line)
     
    def IsClosed(self):
        return self.segs[-1].a==self.points[-1] and self.segs[-1].b==self.points[0]
    
    def __getitem__(self, index):
        return self.points[index]
    
    @property
    def IsClosedLike(self):
        return self.IsClosed() or self.Dist(self.points[0], self.points[-1]) < self.avgDist + 2*self.stdDist
    
    @staticmethod
    def ListMidIndices(lis, denominator):
        '''gives the indices for the middle fraction of a list'''
        step = len(lis)
        return int((step/denominator)*(denominator/2)), int((step/denominator)*(denominator/2 + 1))
    
    @staticmethod
    def Dist(a, b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)    
       
class ContourShape(Shape):
    close = True
    
    def __init__(self, levelset):
        x,y = np.mgrid[:levelset.shape[0],:levelset.shape[1]]
        c = cntr.Cntr(x,y,levelset)
        res = c.trace(.5)
        super(self.__class__, self).__init__(res[0][...,::-1])
    
    def PlotGrid(self, dims, background):
        super(self.__class__, self).PlotGrid(dims, background, transpose=False)

class CenterShape(Shape):
    def __init__(self, levelset):
        x,y = [],[]
#         SetBackground(skeletonize(levelset))
        it = np.nditer(skeletonize(levelset), flags=['multi_index'])
        while not it.finished:
            if it[0]:
                x.append(it.multi_index[1])
                y.append(it.multi_index[0])
            it.iternext()
        x.reverse()
        y.reverse()
        zipped = zip(x,y)
        zipped.sort()
        centerDict = {}
        for xy in zipped:
            if not xy[0] in centerDict:
                centerDict[xy[0]] = xy[1]
        centerLine = centerDict.items()
        centerLine.sort()
        centerLine = zip(*centerLine)
        cLMI = centerLineMidIndices = Shape.ListMidIndices(centerLine[0], denominator=1.8)
        tck = interpolate.splrep(centerLine[0][cLMI[0]:cLMI[1]], centerLine[1][cLMI[0]:cLMI[1]], k=1, s=100)
        xnew = np.arange(-10,np.max(levelset.shape)+10,(np.max(levelset.shape)+20)/200.0)
        ynew = interpolate.splev(xnew, tck, der=0)
        super(self.__class__, self).__init__(zip(xnew,ynew))
    
    def GetGrid(self, dims):
        return super(self.__class__, self).GetGrid(dims, transpose=True)
        
class PoleShape(Shape):
    segLen = 1
    
    def __init__(self, points):
        points.sort(key=lambda x:np.sqrt(np.dot(x,x)))
        super(self.__class__, self).__init__(points)

class RibsShape(Shape):
    segLen = 1
    
    def __init__(self, contourShape, centerShape, poleShape, ribSpacing=1):
        # find the explicit points on the center spline closest to the pole points
        poleCenterInterIndexes = poleShape.GetClosest(centerShape)
        poleCenterInterIndexes.sort(key=lambda x:x[1])
        splineInCellLength = centerShape.GetLength(poleCenterInterIndexes[0][1], poleCenterInterIndexes[1][1])
        ribCount = int(splineInCellLength/ribSpacing) - 1
        ribSep = splineInCellLength/float(ribCount)
        ribCen, ribSlope = [],[]
        # walk along the centerline and place ribs
        distTravelled = 0
        sI = poleCenterInterIndexes[0][1] #sI -> splineIndex
        for i in range(ribCount):
            while distTravelled < ribSep:
                distTravelled+=Shape.Dist(centerShape[sI], centerShape[sI+1])
                sI+=1
            moveBack = (distTravelled - ribSep)*((centerShape[sI-1] - centerShape[sI])/np.sqrt(np.dot((centerShape[sI-1] - centerShape[sI]), (centerShape[sI-1] - centerShape[sI]))))
            ribCen.append(centerShape[sI] + moveBack)
            ribSlope.append(-(centerShape[sI][0]-centerShape[sI-1][0])/(centerShape[sI][1]-centerShape[sI-1][1]))
            distTravelled = (distTravelled - ribSep)
        if len(ribCen)>0:
            riblongs = []
            # draw long ribs
            for i,rc in enumerate(ribCen):
                xa,ya = rc + np.array((40,40*ribSlope[i]))
                xb,yb = rc - np.array((40,40*ribSlope[i]))
                riblongs.append(Shape(((xa,ya),(xb,yb))))
            
            # crop the ribs so that they don't extend beyond the cell contour
            ribs = []
            for riblong in riblongs:
                inter = riblong.GetIntersect(contourShape)
                ribs+=inter
            super(self.__lass__, self).__init__(ribs)
    
    def AvgMidRibLen(self):
        midIndices = Shape.ListMidIndices(self.segs, denominator=5)
        return np.mean([seg.length for seg in self.segs[midIndices[0]:midIndices[1]]])

class CapsuleShape(Shape):
    close = True
    resolution = 500
    
    def __init__(self, poleShape, ribsShape):
        #code to make/fit/show capsule
        radius = ribsShape.AvgMidRibLen()/2.0
        cens = []
        capPoints = []
        for boo in (False, True):
            cenVec = radius*poleShape.segs[0].GetUnitVec(reverse=boo)
            cens.append(poleShape.points[1-boo] - cenVec)
            cen = poleShape.points[1-boo] - cenVec
            seg = Seg(cen, poleShape.points[1-boo])
            seg.Rotate(-math.pi/2)
            capPoints.append(seg.b)
            capPoints.append(seg.b)
            angStep = math.pi/500
            for i in range(500):
                seg.Rotate(angStep)
                capPoints.append(seg.b)
        super(self.__class__, self).__init__(capPoints)
        
if __name__=='__main__':
    shapea = Shape(((-15,-15),(20,20)))
    shapeb = Shape(((20,-20),(-15,15)))
    inter = shapea.GetIntersect(shapeb)
    print inter