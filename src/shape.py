'''
Created on Jun 24, 2014

@author: tel
'''
from copy import deepcopy
import itertools
import math
from matplotlib import _cntr as cntr
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as __plt
import numpy as np
from src.line import Line
from scipy import interpolate
from skimage.morphology import skeletonize

_plt = __plt
_fig = _plt.figure(1)
_ax0 = _plt.subplot(gridspec.GridSpec(1,1,wspace=0.15,hspace=0.07)[0,0])

def AutoscaleAxes():
    _ax0.relim()
    _ax0.autoscale_view(True,True,True)

def SetBackground(background, smooth=True):
    if smooth:
        _plt.imshow(background, cmap=_plt.cm.gray, origin='lower')
    else:
        _plt.pcolormesh(background, cmap=_plt.cm.gray)

def SavePlot(outfile):
    _fig.set_size_inches(12,12)
    _plt.savefig(outfile, bbox_inches='tight')

def ShowPlot():
    _plt.show()

class Shape(object):
    close = False
    # lineLen is the length (in points) of the walk that makes up one of the Shape's lines.
    # -1=inf (closed shape), 0=point, 1=line segment, 2=two joined segments, etc.
    lineLen = -1
    
    def __init__(self, points):
        self.points = np.array(points)
        self.GenLines()
        self.CalcDist()

    @property
    def ax0(self):
        return _ax0
    
    @property
    def IsClosedLike(self):
        return self.IsClosed() or self.Dist(self.points[0], self.points[-1]) < self.avgDist + 2*self.stdDist
    
    @property
    def fig(self):
        return _fig
    
    @property
    def plt(self):
        return _plt
    
    def CalcDist(self):
        if self.lineLen==0 or len(self.lines)==0:
            self.maxDist = 0
            self.avgDist = 0
            self.stdDist = 0
        else:
            dists = []
            for line in self.lines:
                dists.append(line.length)
#             for pa, pb in zip(self.points[:-1], self.points[1:]):
#                 dists.append(self.Dist(pa,pb))
            self.maxDist = np.max(dists)
            self.avgDist = np.mean(dists)
            self.stdDist = np.std(dists)
    
    def Copy(self):
        return deepcopy(self)
    
    def GenLines(self):
        self.lines = []
        if self.lineLen < 0:
            for pa,pb in zip(self.points[:-1], self.points[1:]):
                self.lines.append(Line(pa, pb))
            if self.close:
                self.lines.append(Line(self.points[-1], self.points[0]))
        elif self.lineLen==0:
            pass
        else:
            for i in range(len(self.points))[::self.lineLen+1]:
                for j in range(self.lineLen):
                    self.lines.append(Line(*self.points[i+j:i+j+2]))
        
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
        for sLine,oLine in ((sLine,oLine) for sLine in self.lines for oLine in other.lines):
            intersect = sLine.GetIntersect(oLine)
            if type(intersect)!=type(False):
                ret.append(intersect)
        return ret
    
    def GetGrid(self, dims, transpose=False):
        flipBit = 1 - 2*transpose
        ret = np.zeros(dims[::flipBit])
        for seg in self.lines:
            try:
                for p in (np.floor(p) for p in seg):
                    if (0<=p[0]<dims[::flipBit][0]) and (0<=p[1]<=dims[::flipBit][1]):
                        ret.__setitem__(tuple(p[::flipBit].tolist()), 1)
            except IndexError:
                pass
#             ret = np.logical_or(ret, interpolate.griddata(line + [line[1] + np.array((1,0))] + [line[0] + np.array((1,0))] , (True,)*4, tuple(np.mgrid[:shape[0],:shape[1]]), fill_value=False))
        return ret
    
    def GetLength(self, i1, i2):
        '''gives the length of the line segments between points with index i1 and i2'''
        ret = 0
        i1,i2 = sorted((i1,i2))
        for line in self.lines[i1:i2]:
            ret+=line.length
        return ret
    
    def PlotConnectedPoints(self, c='b', lw=1, alpha=1):
        '''use PlotConnectedPoints instead of PlotSegs if you use alpha!=1 and want to maintain proper transparency throughout your plotted curves'''
        if c=='cycle':
            c=self.GetColor()
        (x,y) = zip(*self.points)[:2]
        if self.close:
            x = list(x) + [self.points[0][0]]
            y = list(y) + [self.points[0][1]]
        self.ax0.plot(x, y, c=c, lw=lw, alpha=alpha)
    
    def PlotGrid(self, dims, background=None, transpose=False):
        flipBit = 1 - 2*transpose
        for lines in self.lines:
            self.plt.plot(*zip(*lines)[::flipBit], c='r')
#         plt.axis([0,dims[0],0,dims[1]])
        if background is not None:
            self.plt.imshow(background, cmap=_plt.cm.gray, origin='lower')
        self.plt.show()
    
    def PlotPoints(self, radius=.7, fc='g'):
        if fc=='cycle':
            fc=self.GetColor()
        for point in self.points:
            circle = _plt.Circle(point, radius, fc=fc)
            self.ax0.add_patch(circle)
    
    def PlotSegs(self, c='b', lw=1, alpha=1):
        if c=='cycle':
            c=self.GetColor()
        for lines in self.lines:
            line = _plt.Line2D(*zip(*lines)[:2], c=c, lw=lw, alpha=alpha)
            self.ax0.add_line(line)
     
    def IsClosed(self):
        return self.lines[-1].a==self.points[-1] and self.lines[-1].b==self.points[0]
    
    def Overlay(self, seg0, seg1):
        '''transforms all of the points and lines in self by the transformation matrix that overlays seg0 on seg1.
        currently the section that deals with points only works correctly for shapes with self.lineLen = -1'''
        transMat = Line.OverlaySegMat(seg0, seg1)
        # transform points
        homoPoints = np.zeros((self.points.shape[0],4))
        homoPoints[:,:2] = self.points[:,:2]
        homoPoints[:,3] = 1
        self.points = transMat.dot(homoPoints.T).T[:,:2]
        # regenerate segments from the newly transformed points
        self.GenLines()
        
    def OverlayOnCell(self, capsuleShape):
        self.Overlay(capsuleShape.cenLines[0], Line((0,0),(1,0)))
    
    def __getitem__(self, index):
        return self.points[index]
    
    @classmethod
    def GetColor(cls, cmap=cm.rainbow, cycleLen=20):
        '''gets the next color in a color map. has a persistent state, and will wrap around if it reaches the end'''
        try:
            if cmap.name==cls.colorCycleCmap.name and cycleLen==cls.colorCycleLen:
                return cls.colorCycle.next()
            else:
                cls.colorCycleCmap = cmap
                cls.colorCycleLen = cycleLen
                cls.colorCycle = itertools.cycle(cls.colorCycleCmap(np.linspace(0, 1, cls.colorCycleLen)))
                return cls.colorCycle.next()
        except AttributeError:
            cls.colorCycleCmap = cmap
            cls.colorCycleLen = cycleLen
            cls.colorCycle = itertools.cycle(cls.colorCycleCmap(np.linspace(0, 1, cls.colorCycleLen)))
            return cls.colorCycle.next()
    
    @classmethod
    def ResetColor(cls, cmap=cm.rainbow, cycleLen=20):
        cls.colorCycleCmap = getattr(cls, 'colorCycleCmap', cmap)
        cls.colorCycleLen = getattr(cls, 'colorCycleLen', cycleLen)
        cls.colorCycle = itertools.cycle(cls.colorCycleCmap(np.linspace(0, 1, cls.colorCycleLen)))
    
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
        cLMI = centerLineMidIndices = Shape.ListMidIndices(centerLine[0], denominator=1)
        tck = interpolate.splrep(centerLine[0][cLMI[0]:cLMI[1]], centerLine[1][cLMI[0]:cLMI[1]], k=1, s=100)
        xnew = np.arange(0-.5,np.max(levelset.shape)-.5,(np.max(levelset.shape))/201.0)
        ynew = interpolate.splev(xnew, tck, der=0)
        super(self.__class__, self).__init__(zip(xnew,ynew))
    
    def GetGrid(self, dims):
        return super(self.__class__, self).GetGrid(dims, transpose=True)
        
class PoleShape(Shape):
    lineLen = 1
    
    def __init__(self, contourShape, centerShape):
        points = contourShape.GetIntersect(centerShape)
        points.sort(key=lambda x:np.sqrt(np.dot(x,x)))
        super(self.__class__, self).__init__(points)

class RibsShape(Shape):
    lineLen = 2
    
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
                riblongs.append(Shape(((xa,ya), rc, (xb,yb))))
            
            # crop the ribs so that they don't extend beyond the cell contour
            ribs = []
            for riblong in riblongs:
                inter = riblong.GetIntersect(contourShape)
                ribs+= [inter[0], riblong[1], inter[1]]
            super(self.__class__, self).__init__(ribs)
    
    def AvgMidRibLen(self):
        '''each rib is made out of two consecutive line segments, so when finding the length of a rib we need to sum the two'''
        midIndices = Shape.ListMidIndices(self.lines, denominator=5)
        return np.mean([lineOne.length+lineTwo.length for lineOne,lineTwo in zip(self.lines[midIndices[0]:midIndices[1]:2], self.lines[midIndices[0]+1:midIndices[1]:2])])
    
    def Quality(self, other):
        if np.allclose(self.lines[0].b, other.lines[0].b) and len(self.lines)==len(other.lines):
            return np.sum([(s.length - o.length)**2 for s,o in zip(self.lines, other.lines)])
        else:
            return 9999

class CapsuleShape(Shape):
    close = True
    resolution = 500
    
    def __init__(self, poleShape, ribsShape):
        #code to make/fit/show capsule
        capPoints = []
        # cenLines is used later to help with the trajectory transforms
        self.cenLines = []
        self.radius = ribsShape.AvgMidRibLen()/2.0
        for boo in (False, True):
            cenVec = (self.radius*poleShape.lines[0].GetUnitVec(reverse=boo))
            cen = poleShape.points[boo] + cenVec
            line = Line(cen, poleShape.points[boo])
            self.cenLines.append(Line(cen, cen + cenVec))
            line.Rotate(-math.pi/2)
            capPoints.append(line.b)
            angStep = math.pi/500
            for i in range(500):
                line.Rotate(angStep)
                capPoints.append(line.b)
        super(self.__class__, self).__init__(capPoints)

class TrajectoryShape(Shape):
    def __init__(self, trajPoints):
        super(self.__class__, self).__init__(trajPoints)
            
if __name__=='__main__':
    shapea = Shape(((-15,-15),(20,20)))
    shapeb = Shape(((20,-20),(-15,15)))
    inter = shapea.GetIntersect(shapeb)
    print inter