'''
Created on Sep 15, 2014

@author: tel
'''
import math
import numpy as np
from src import exception
from transmat import rotation_matrix, translation_matrix

class Line(object):
    def __init__(self, *args):
        self.a = np.array((0,0,0,1), dtype=np.float)
        self.b = np.array((0,0,0,1), dtype=np.float)
        if len(args)==2 and hasattr(args[0], "__getitem__") and hasattr(args[1], "__getitem__"):
            self.a[:len(args[0])] = args[0]
            self.b[:len(args[1])] = args[1]
        elif len(args>=2):
            try:
                args = map(float, args)
            except ValueError or TypeError:
                raise exception.SegInitError('one of the four args passed to Line was not a number. Your args were %s' % args)
            self.a[:len(args)/2] = np.array(args[:len(args)/2])
            self.b[:len(args)/2] = np.array(args[len(args)/2:])
        else:
            raise exception.SegInitError('the arguments passed to Line should either be two lists of floats ((xa,ya),(xb,yb)) or four to eight floats (xa,yb,xb,yb). Your args were %s' % args)
        self.CalcLength()
    
    def CalcLength(self):
        self.length = self.GetLength()
        
    def Copy(self):
        return self.__class__(self.a, self.b)
    
    def GetIntersect(self, other):
        '''test if two Line objects intersect. Returns an array of coordinates if they do, and False otherwise. lord help you if they overlap instead. adapted from http://stackoverflow.com/q/563198/425458
        if one segment of an (apparently) straight line ends on a line you're testing for intersections, and another segment of the first line then starts where the last ended, then both will be regarded as intersecting'''
        rcrosss = np.cross((self.b[:3] - self.a[:3]), (other.b[:3] - other.a[:3]))
        if not np.any(rcrosss):
            return False
        t = (np.cross((other.a[:3] - self.a[:3]),(other.b[:3] - other.a[:3]))/rcrosss)[2]
        u = (np.cross((other.a[:3] - self.a[:3]),(self.b[:3] - self.a[:3]))/rcrosss)[2]
        if np.all(0<=t<=1) and np.all(0<=u<=1):
            return self.a + t*(self.b - self.a)
        else:
            return False
        
    def GetLength(self):
        return np.sqrt(np.sum((self.b - self.a)**2))

    def GetSlope(self):
        return (self.b[0] - self.a[0])/(self.b[1] - self.a[1])

    def GetUnitVec(self, reverse=False):
        if reverse:
            diff = self.a - self.b
        else:
            diff = self.b - self.a
        return (diff/np.sqrt(np.dot(diff, diff)))
    
    def Reversed(self):
        return Line(self.b, self.a)
    
    def Rotate(self, ang, pnt=None):
        if pnt==None:
            pnt = self.a
        self.b = self.__class__.RotatePointMat(ang, pnt[:3]).dot(self.b.T)
        
    def AffTrans(self, mat):
        '''takes a 4x4 affine-transformation matrix and uses it to alter self in place'''
        self.a,self.b = mat.dot(self.a),mat.dot(self.b)
    
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
    
    @staticmethod
    def RotatePointMat(ang, pnt, seg1=None, seg2=None):
        '''returns 4x4 transformation matrix that will rotate by ang (in radians) about pnt. 
        line segments seg1 and seg2 may optionally be specified in order to fix the axis of rotation as their cross product.
        each seg is a line segment represented by a list of two points. otherwise, rotation occurs about the z axis'''
        if seg1!=None and seg2!=None:
            vec1 = np.array(seg1[1]) - np.array(seg1[0])
            vec2 = np.array(seg2[1]) - np.array(seg2[0])
            if np.allclose(vec1,vec2):
                axis = (0,0,-1)
            else:
                axis = np.cross(vec1, vec2)
        else:
            axis = (0,0,-1)
        return rotation_matrix(ang, axis, point=pnt)
    
    @staticmethod
    def OverlaySegMat(seg0, seg1):
        '''returns 4x4 transformation matrix that will overlay seg0 on seg1. each seg is a line segment represented by a list of two points'''
        vec0 = (np.array(seg0[1]) - np.array(seg0[0]))[:3]
        vec1 = (np.array(seg1[1]) - np.array(seg1[0]))[:3]
        axis = np.cross(vec0, vec1)
        if np.allclose([0,0,0],axis):
            axis = (0,0,1)
        if np.allclose(vec0,vec1):
            ang = 0
        elif np.allclose(vec0,-vec1):
            ang = math.pi
        else:
            ang = np.arccos(np.dot(vec0, vec1)/(np.sqrt(np.dot(vec0, vec0))*np.sqrt(np.dot(vec1, vec1))))
#        if np.isnan(ang):
#            ang = math.pi
        trans = np.array(seg1[0]) - np.array(seg0[0])
        return translation_matrix(trans).dot(rotation_matrix(ang, axis, point=seg0[0]))