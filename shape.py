'''
Created on Jun 24, 2014

@author: tel
'''
import math, itertools
import numpy as np
from matplotlib import pyplot as __plt
import matplotlib.gridspec as gridspec
from scipy import interpolate
from skimage.morphology import skeletonize
import exception
from matplotlib import _cntr as cntr

_plt = __plt
_fig = _plt.figure(1)
_ax0 = _plt.subplot(gridspec.GridSpec(1,1,wspace=0.15,hspace=0.07)[0,0])

def SetBackground(background):
    _plt.imshow(background, cmap=_plt.cm.gray, origin='lower')
    
def ShowPlot():
    _plt.show()

class SharedPltMetaclass(type): 
    def __new__(cls, clsname, bases, dct):
        figx = 12
        figy = 10
        #fig = plt.figure(figsize=(figx*len(args)/2,figy))
        dct['_fig'] = plt.figure(1)
        gs = gridspec.GridSpec(1,1)
        gs.update(wspace=0.15, hspace=0.07)
        dct['_plt'] = plt
        dct['_ax0'] = plt.subplot(gs[0, 0])

        #return type.__new__(cls, clsname, bases, dct)
        #return super(cls.__class__, cls).__new__(cls, clsname, bases, dct)
        return super(SharedPltMetaclass, cls).__new__(cls, clsname, bases, dct)

class Shape(object):
#     __metaclass__ = SharedPltMetaclass
    
#     _plt = plt
#     _fig = plt.figure(1)
#     _ax0 = plt.subplot(gridspec.GridSpec(1,1,wspace=0.15,hspace=0.07)[0,0])
    
    def __init__(self, points):
        self.points = np.array(points)
        self.CalcDist()
        self.GenSegs()
    
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
        dists = []
        for pa, pb in zip(self.points[:-1], self.points[1:]):
            dists.append(self.Dist(pa,pb))
        self.maxDist = np.max(dists)
        self.avgDist = np.mean(dists)
        self.stdDist = np.std(dists)
        
    def GenSegs(self, close=False, segLen=-1):
        '''
        segLen is the length of the walk that makes up one of the Shape's segs.
        -1=inf (closed shape), 0=point, 1=line segment, 2=two joined segments, etc.
        '''
        self.close = close
        self.segLen = segLen
        self.segs = []
        if self.segLen < -1:
            for pa,pb in zip(self.points[:-1], self.points[1:]):
                self.segs.append(Seg(pa, pb))
            if self.close:
                self.segs.append(Seg(self.points[-1], self.points[0]))
        elif self.segLen==0:
            pass
        else:
            for i in range(len(self.points))[::self.segLen+1]:
                for j in range(self.segLen):
                    self.segs.append(Seg(*self.points[i+j:i+j+1]))
        
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
    
    def PlotPoints(self, radius=.7):
        for point in self.points:
            circle = _plt.Circle(point, radius)
            self.ax0.add_patch(circle)
    
    def PlotSegs(self):
        for seg in self.segs:
            line = _plt.Line2D(*zip(*seg))
            self.ax0.add_line(line)
    
    def PlotMyGrid(self, dims):
        for seg in self.segs:
            self.plt.plot(*zip(*seg), c='r')
#         self.__class__.plt.show()
     
    def IsClosed(self):
        return self.segs[-1].a==self.points[-1] and self.segs[-1].b==self.points[0]
    
    def __getitem__(self, index):
        return self.points[index]
    
    @property
    def IsClosedLike(self):
        return self.IsClosed() or self.Dist(self.points[0], self.points[-1]) < self.avgDist + 2*self.stdDist
    
    @staticmethod
    def Dist(a, b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    
class Seg(object):
    def __init__(self, *args):
        if len(args)==2 and hasattr(args[0], "__getitem__") and hasattr(args[1], "__getitem__"):
            self.a = np.array(args[0])
            self.b = np.array(args[1])
        elif len(args==4):
            try:
                args = map(float, args)
            except ValueError or TypeError:
                raise exception.SegInitError('one of the four args passed to Seg was not a number. Your args were %s' % args)
            self.a = np.array(args[:2])
            self.b = np.array(args[2:])
        else:
            raise exception.SegInitError('the arguments passed to Seg should either be two lists of two floats ((xa,ya),(xb,yb)) or four floats (xa,yb,xb,yb). Your args were %s' % args)
        self.CalcLength()
    
    def CalcLength(self):
        self.length = Shape.Dist(self.a, self.b)
    
    def GetIntersect(self, other):
        '''test if two Seg objects intersect. Returns an array of coordinates if they do, and False otherwise. lord help you if they overlap instead. adapted from http://stackoverflow.com/q/563198/425458'''
        rcrosss = float(np.cross((self.b - self.a), (other.b - other.a)))
        if not rcrosss:
            return False
        t = np.cross((other.a - self.a),((other.b - other.a)))/rcrosss
        u = np.cross((other.a - self.a),(self.b - self.a))/rcrosss
        if (0<=t<=1) and (0<=u<=1):
            return self.a + t*(self.b - self.a)
        else:
            return False
        
    def __getitem__(self, i):
        if i==0:
            return self.a
        elif i==1:
            return self.b
        else:
            raise exception.SegAccessError
        
    def __setitem__(self, i, val):
        if i==0:
            self.a = val
        elif i==1:
            self.b = val
        else:
            raise exception.SegAccessError
        
    def __iter__(self):
        for i in range(2):
            yield self[i]
    
    def __repr__(self):
        return str([p for p in self])
    
    def __str__(self):
        return str(self.__repr__())
    
    def __add__(self, other):
        return self.__repr__() + other
        
class ContourShape(Shape):
    def __init__(self, levelset):
        x,y = np.mgrid[:levelset.shape[0],:levelset.shape[1]]
        c = cntr.Cntr(x,y,levelset)
        res = c.trace(.5)
        super(self.__class__, self).__init__(res[0][...,::-1], close=True)
    
    def GenSegs(self):
        super(self.__class__, self).GenSegs(close=True)
    
    def PlotGrid(self, dims, background):
        super(self.__class__, self).PlotGrid(dims, background, transpose=False)
        
    def PlotGridShared(self, dims, background):
        super(self.__class__, self).PlotGridShared(dims, background, transpose=False)

class CenterShape(Shape):
    def __init__(self, levelset):
        x,y = [],[]
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
        centerdict = {}
        for xy in zipped:
            if not xy[0] in centerdict:
                centerdict[xy[0]] = xy[1]
        centerline = centerdict.items()
        centerline.sort()
        tck = interpolate.splrep(zip(*centerline[len(centerline)/4:-len(centerline)/4])[0], zip(*centerline[len(centerline)/4:-len(centerline)/4])[1],k=1,s=100)
        xnew = np.arange(-10,np.max(levelset.shape)+10,(np.max(levelset.shape)+20)/200.0)
        ynew = interpolate.splev(xnew, tck, der=0)
        super(self.__class__, self).__init__(zip(xnew,ynew))
    
    def GetGrid(self, dims):
        return super(self.__class__, self).GetGrid(dims, transpose=True)
        
class PoleShape(Shape):
    def __init__(self, points):
        super(self.__class__, self).__init__(points)
        
    def GenSegs(self):
        super(self.__class__, self).GenSegs(segLen=0)

    def GetGrid(self, dims):
        return super(self.__class__, self).GetGrid(dims, transpose=False)

class RibShape(Shape):
    def __init__(self, points):
        super(self.__class__, self).__init__(points)
        
    def GetGrid(self, dims):
        return super(self.__class__, self).GetGrid(dims, transpose=True)

class RibsShape(object):    
    def __init__(self, contourShape, centerShape, poleShape, ribSpacing=1):
        #find the explicit points on the center spline closest to the pole points
        poleCenterInterIndexes = poleShape.GetClosest(centerShape)
        splineInCellLength = centerShape.GetLength(poleCenterInterIndexes[0][1], poleCenterInterIndexes[1][1])
        ribCount = int(splineInCellLength/ribSpacing) - 1
        ribSep = splineInCellLength/float(ribCount)
        ribCen, ribSlope = [],[]
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
            riblongplots = []
            #draw long ribs
            for i,rc in enumerate(ribCen):
                xa,ya = rc + np.array((40,40*ribSlope[i]))
                xb,yb = rc - np.array((40,40*ribSlope[i]))
                riblongs.append(Shape(((xa,ya),(xb,yb))))
#                 riblongplots.append(ax1.plot((xa,xb), (ya,yb), c='y'))
#             ax1.imshow(background + 1000*skeletonize(msnake.levelset), cmap=ppl.cm.gray)
#             ax_u.set_data(msnake.levelset)
#             fig.canvas.draw()
            
            #crop the ribs so that they don't extend beyond the cell contour
            self.ribList = []
            for riblong,riblongplot in zip(riblongs,riblongplots):
                inter = riblong.GetIntersect(contourShape)
#                 riblongplot.pop(0).remove()
                self.ribList.append(RibShape(inter))
#                 ribx = (inter[0][0], inter[1][0])
#                 riby = (inter[0][1], inter[1][1])
#                 ribplots.append(ax1.plot(ribx, riby, c='y'))
#             ax1.imshow(background + 1000*skeletonize(msnake.levelset), cmap=ppl.cm.gray)
#             ax_u.set_data(msnake.levelset)
#             fig.canvas.draw()
    def GenSegs(self):
        super(self.__class__, self).GenSegs(segLen=1)

class CapsuleShape(Shape):
    def __init__(self, points):
        super(self.__class__, self).__init__(points, close=True)

if __name__=='__main__':
    shapea = Shape(((-15,-15),(20,20)))
    shapeb = Shape(((20,-20),(-15,15)))
    inter = shapea.GetIntersect(shapeb)
    print inter