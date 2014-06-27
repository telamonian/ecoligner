'''
Created on Jun 24, 2014

@author: tel
'''
import numpy as np
import exception

class Shape(object):
    def __init__(self, points):
        self.points = points
        self.CalcDist()
        self.genSegs()
    
    def genSegs(self):
        self.segs = []
        for pa,pb in zip(self.points[:-1], self.points[1:]):
            self.segs.append(Seg(pa, pb))
        if self.IsClosed:
            self.segs.append(Seg(self.points[-1], self.points[0]))
            
    def CalcDist(self):
        dists = []
        for pa, pb in zip(self.points[:-1], self.points[1:]):
            dists.append(self.Dist(pa,pb))
        self.maxDist = np.max(dists)
        self.avgDist = np.mean(dists)
        self.stdDist = np.std(dists)
    
    def GetIntersect(self, other):
        ret = []
        for sSeg,oSeg in ((sSeg,oSeg) for sSeg in self.segs for oSeg in other.segs):
            intersect = sSeg.GetIntersect(oSeg)
            if type(intersect)!=type(False):
                ret.append(intersect)
        return ret

    @property
    def IsClosed(self):
        return self.Dist(self.points[0], self.points[-1]) < self.avgDist + 2*self.stdDist
    
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
    
    #test if two Seg objects intersect. Returns an array of coordinates if they do, and False otherwise. lord help you if they overlap instead. adapted from http://stackoverflow.com/q/563198/425458
    def GetIntersect(self, other):
        rcrosss = float(np.cross((self.b - self.a), (other.b - other.a)))
        if not rcrosss:
            return False
        t = np.cross((other.a - self.a),((other.b - other.a)))/rcrosss
        u = np.cross((other.a - self.a),(self.b - self.a))/rcrosss
        if (0<=t<=1) and (0<=u<=1):
            return self.a + t*(self.b - self.a)
        else:
            return False

if __name__=='__main__':
    shapea = Shape(((-15,-15),(20,20)))
    shapeb = Shape(((20,-20),(-15,15)))
    inter = shapea.GetIntersect(shapeb)
    print inter
    print 'hey'