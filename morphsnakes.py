# -*- coding: utf-8 -*-

"""
morphsnakes
===========

This is a Python implementation of the algorithms introduced in the paper

  Márquez-Neila, P., Baumela, L., Álvarez, L., "A morphological approach
  to curvature-based evolution of curves and surfaces". IEEE Transactions
  on Pattern Analysis and Machine Intelligence (PAMI), 2013.

This implementation is intended to be as brief, understandable and self-contained
as possible. It does not include any enhancement to make it fast or efficient.

Any practical implementation of this algorithm should work only over the
neighbor pixels of the 0.5-levelset, not over all the embedding function,
and perhaps should feature multi-threading or GPU capabilities.

The classes MorphGAC and MorphACWE provide most of the functionality of this
module. They implement the Morphological Geodesic Active Contours and the
Morphological Active Contours without Edges, respectively. See the
aforementioned paper for full details.

See cocoligny for examples of usage.
"""

__author__ = "P. Márquez Neila <p.mneila@upm.es>"

from itertools import cycle

import math, itertools
import numpy as np
import shape
from scipy import ndimage, interpolate
from skimage.morphology import skeletonize
from scipy.ndimage import binary_dilation, binary_erosion, \
                        gaussian_filter, gaussian_gradient_magnitude

class fcycle(object):
    
    def __init__(self, iterable):
        """Call functions from the iterable each time it is called."""
        self.funcs = cycle(iterable)
    
    def __call__(self, *args, **kwargs):
        f = self.funcs.next()
        return f(*args, **kwargs)
    

# SI and IS operators for 2D and 3D.
_P2 = [np.eye(3), np.array([[0,1,0]]*3), np.flipud(np.eye(3)), np.rot90([[0,1,0]]*3)]
_P3 = [np.zeros((3,3,3)) for i in xrange(9)]

_P3[0][:,:,1] = 1
_P3[1][:,1,:] = 1
_P3[2][1,:,:] = 1
_P3[3][:,[0,1,2],[0,1,2]] = 1
_P3[4][:,[0,1,2],[2,1,0]] = 1
_P3[5][[0,1,2],:,[0,1,2]] = 1
_P3[6][[0,1,2],:,[2,1,0]] = 1
_P3[7][[0,1,2],[0,1,2],:] = 1
_P3[8][[0,1,2],[2,1,0],:] = 1

_aux = np.zeros((0))
def SI(u):
    """SI operator."""
    global _aux
    if np.ndim(u) == 2:
        P = _P2
    elif np.ndim(u) == 3:
        P = _P3
    else:
        raise ValueError, "u has an invalid number of dimensions (should be 2 or 3)"
    
    if u.shape != _aux.shape[1:]:
        _aux = np.zeros((len(P),) + u.shape)
    
    for i in xrange(len(P)):
        _aux[i] = binary_erosion(u, P[i])
    
    return _aux.max(0)

def IS(u):
    """IS operator."""
    global _aux
    if np.ndim(u) == 2:
        P = _P2
    elif np.ndim(u) == 3:
        P = _P3
    else:
        raise ValueError, "u has an invalid number of dimensions (should be 2 or 3)"
    
    if u.shape != _aux.shape[1:]:
        _aux = np.zeros((len(P),) + u.shape)
    
    for i in xrange(len(P)):
        _aux[i] = binary_dilation(u, P[i])
    
    return _aux.min(0)

# SIoIS operator.
SIoIS = lambda u: SI(IS(u))
ISoSI = lambda u: IS(SI(u))
curvop = fcycle([SIoIS, ISoSI])

# Stopping factors (function g(I) in the paper).
def gborders(img, alpha=1.0, sigma=1.0):
    """Stopping criterion for image borders."""
    # The norm of the gradient.
    gradnorm = gaussian_gradient_magnitude(img, sigma, mode='constant')
    return 1.0/np.sqrt(1.0 + alpha*gradnorm)

def glines(img, sigma=1.0):
    """Stopping criterion for image black lines."""
    return gaussian_filter(img, sigma)

class MorphACWE(object):
    """Morphological ACWE based on the Chan-Vese energy functional."""
    
    def __init__(self, data, smoothing=1, lambda1=1, lambda2=1):
        """Create a Morphological ACWE solver.
        
        Parameters
        ----------
        data : ndarray
            The image data.
        smoothing : scalar
            The number of repetitions of the smoothing step (the
            curv operator) in each iteration. In other terms,
            this is the strength of the smoothing. This is the
            parameter µ.
        lambda1, lambda2 : scalars
            Relative importance of the inside pixels (lambda1)
            against the outside pixels (lambda2).
        """
        self._u = None
        self.smoothing = smoothing
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
        self.data = data
    
    def set_levelset(self, u):
        self._u = np.double(u)
        self._u[u>0] = 1
        self._u[u<=0] = 0
    
    levelset = property(lambda self: self._u,
                        set_levelset,
                        doc="The level set embedding function (u).")
    
    def step(self):
        """Perform a single step of the morphological Chan-Vese evolution."""
        # Assign attributes to local variables for convenience.
        u = self._u
        
        if u is None:
            raise ValueError, "the levelset function is not set (use set_levelset)"
        
        data = self.data
        
        # Determine c0 and c1.
        inside = u>0
        outside = u<=0
        c0 = data[outside].sum() / float(outside.sum())
        c1 = data[inside].sum() / float(inside.sum())
        
        # Image attachment.
        dres = np.array(np.gradient(u))
        abs_dres = np.abs(dres).sum(0)
        #aux = abs_dres * (c0 - c1) * (c0 + c1 - 2*data)
        aux = abs_dres * (self.lambda1*(data - c1)**2 - self.lambda2*(data - c0)**2)
        
        res = np.copy(u)
        res[aux < 0] = 1
        res[aux > 0] = 0
        
        # Smoothing.
        for i in xrange(self.smoothing):
            res = curvop(res)
        
        self._u = res
    
    def run(self, iterations):
        """Run several iterations of the morphological Chan-Vese method."""
        for i in xrange(iterations):
            self.step()
    

class MorphGAC(object):
    """Morphological GAC based on the Geodesic Active Contours."""
    
    def __init__(self, data, smoothing=1, threshold=0, balloon=0):
        """Create a Morphological GAC solver.
        
        Parameters
        ----------
        data : array-like
            The stopping criterion g(I). See functions gborders and glines.
        smoothing : scalar
            The number of repetitions of the smoothing step in each
            iteration. This is the parameter µ.
        threshold : scalar
            The threshold that determines which areas are affected
            by the morphological balloon. This is the parameter θ.
        balloon : scalar
            The strength of the morphological balloon. This is the parameter ν.
        """
        self._u = None
        self._v = balloon
        self._theta = threshold
        self.smoothing = smoothing
        
        self.set_data(data)
    
    def set_levelset(self, u):
        self._u = np.double(u)
        self._u[u>0] = 1
        self._u[u<=0] = 0
    
    def set_balloon(self, v):
        self._v = v
        self._update_mask()
    
    def set_threshold(self, theta):
        self._theta = theta
        self._update_mask()
    
    def set_data(self, data):
        self._data = data
        self._ddata = np.gradient(data)
        self._update_mask()
        # The structure element for binary dilation and erosion.
        self.structure = np.ones((3,)*np.ndim(data))
    
    def _update_mask(self):
        """Pre-compute masks for speed."""
        self._threshold_mask = self._data > self._theta
        self._threshold_mask_v = self._data > self._theta/np.abs(self._v)
    
    levelset = property(lambda self: self._u,
                        set_levelset,
                        doc="The level set embedding function (u).")
    data = property(lambda self: self._data,
                        set_data,
                        doc="The data that controls the snake evolution (the image or g(I)).")
    balloon = property(lambda self: self._v,
                        set_balloon,
                        doc="The morphological balloon parameter (ν (nu, not v)).")
    threshold = property(lambda self: self._theta,
                        set_threshold,
                        doc="The threshold value (θ).")
    
    def step(self):
        """Perform a single step of the morphological snake evolution."""
        # Assign attributes to local variables for convenience.
        u = self._u
        gI = self._data
        dgI = self._ddata
        theta = self._theta
        v = self._v
        
        if u is None:
            raise ValueError, "the levelset is not set (use set_levelset)"
        
        res = np.copy(u)
        
        # Balloon.
        if v > 0:
            aux = binary_dilation(u, self.structure)
        elif v < 0:
            aux = binary_erosion(u, self.structure)
        if v!= 0:
            res[self._threshold_mask_v] = aux[self._threshold_mask_v]
        
        # Image attachment.
        aux = np.zeros_like(res)
        dres = np.gradient(res)
        for el1, el2 in zip(dgI, dres):
            aux += el1*el2
        res[aux > 0] = 1
        res[aux < 0] = 0
        
        # Smoothing.
        for i in xrange(self.smoothing):
            res = curvop(res)
        
        self._u = res
    
    def run(self, iterations):
        """Run several iterations of the morphological snakes method."""
        for i in xrange(iterations):
            self.step()
    

def evolve_visual(msnake, levelset=None, num_iters=20, background=None, outfile=None):
    """
    Visual evolution of a morphological snake.
    
    Parameters
    ----------
    msnake : MorphGAC or MorphACWE instance
        The morphological snake solver.
    levelset : array-like, optional
        If given, the levelset of the solver is initialized to this. If not
        given, the evolution will use the levelset already set in msnake.
    num_iters : int, optional
        The number of iterations.
    background : array-like, optional
        If given, background will be shown behind the contours instead of
        msnake.data.
    """
    from matplotlib import pyplot as ppl
    
    if levelset is not None:
        msnake.levelset = levelset
    
    # Prepare the visual environment.
    fig = ppl.gcf()
    fig.clf()
    ax1 = fig.add_subplot(1,2,1)
    if background is None:
        ax1.imshow(msnake.data, cmap=ppl.cm.gray)
    else:
        ax1.imshow(background, cmap=ppl.cm.gray)
    ax1.contour(msnake.levelset, [0.5], colors='r')
    
    ax2 = fig.add_subplot(1,2,2)
    ax_u = ax2.imshow(msnake.levelset)
    ppl.pause(0.001)
    
    spline = None
    intersections = None
    ribplots = []
    capsule = None
    # Iterate.
    for i in xrange(num_iters):
        #save figure if filename is specified
        if outfile:
            ppl.savefig('%s%.4d' % (outfile, i))
        
        # Evolve.
        msnake.step()
        
        # Update figure.
        del ax1.collections[0]
        if spline:
            spline.pop(0).remove()
        if intersections:
            try:
                intersections.remove()
            except ValueError:
                pass
        for ribplot in ribplots:
            ribplot.pop(0).remove()
        if capsule:
            capsule.pop(0).remove()
        ribplots = []
        ax1.contour(msnake.levelset, [0.5], colors='r')
        
        #code to make/show centerspline
        x,y = [],[]
        it = np.nditer(skeletonize(msnake.levelset), flags=['multi_index'])
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
        try:
            tck = interpolate.splrep(zip(*centerline[len(centerline)/4:-len(centerline)/4])[0], zip(*centerline[len(centerline)/4:-len(centerline)/4])[1],k=1,s=100)
            xnew = np.arange(0,40,40.0/200)
            ynew = interpolate.splev(xnew, tck, der=0)
            splineLine = map(np.array, zip(xnew,ynew))
            spline = ax1.plot(xnew, ynew, 'b')
            
            #code to make/show points of intersection of cell boundary and centerspline
            centershape = shape.Shape(zip(xnew,ynew))
            boundshape = shape.Shape(ax1.collections[0].get_paths()[0].vertices)
            inters =  centershape.GetIntersect(boundshape)
            if len(inters)>1:
                intersections = ax1.scatter((inters[0][0],inters[1][0]),(inters[0][1],inters[1][1]), s=100,c='g')
        except TypeError:
            pass
        
        #code to make/crop/show ribs
        if i>15:
            try:
                #find the explicit points on the spline closest to the boundary/spline intersection points
                splineInterIndexes = [9999999,9999999]
                minDist = 9999999
                for i,inter in enumerate(inters):
                    for j,splinePoint in enumerate(splineLine):
                        dist = shape.Shape.Dist(inter, splinePoint)
                        if dist < minDist:
                            minDist = dist
                            splineInterIndexes[i] = j
                    minDist = 9999999
                splineInCellLength = 0
                for splinePointA,splinePointB in zip(splineLine[splineInterIndexes[0]:splineInterIndexes[1]], splineLine[splineInterIndexes[0]+1:splineInterIndexes[1]+1]):
                    splineInCellLength+=shape.Shape.Dist(splinePointA, splinePointB)
                ribCount = int(splineInCellLength) - 1
                ribSep = splineInCellLength/float(ribCount)
                ribCen, ribSlope = [],[]
                distTravelled = 0
                sI = splineInterIndexes[0]  #sI -> splineIndex
                for i in range(ribCount):
                    while distTravelled < ribSep:
                        distTravelled+=shape.Shape.Dist(splineLine[sI], splineLine[sI+1])
                        sI+=1
                    moveBack = (distTravelled - ribSep)*((splineLine[sI-1] - splineLine[sI])/np.sqrt(np.dot((splineLine[sI-1] - splineLine[sI]), (splineLine[sI-1] - splineLine[sI]))))
                    ribCen.append(splineLine[sI] + moveBack)
                    ribSlope.append(-(splineLine[sI][0]-splineLine[sI-1][0])/(splineLine[sI][1]-splineLine[sI-1][1]))
                    distTravelled = (distTravelled - ribSep)
                if len(ribCen)>0:
                    riblongs = []
                    riblongplots = []
                    #draw long ribs
                    for i,rc in enumerate(ribCen):
                        xa,ya = rc + np.array((40,40*ribSlope[i]))
                        xb,yb = rc - np.array((40,40*ribSlope[i]))
                        riblongplots.append(ax1.plot((xa,xb), (ya,yb), c='y'))
                        riblongs.append(shape.Shape(((xa,ya),(xb,yb))))
                    ax1.imshow(background + 1000*skeletonize(msnake.levelset), cmap=ppl.cm.gray)
                    ax_u.set_data(msnake.levelset)
                    fig.canvas.draw()
                    
                    #crop the ribs so that they don't extend beyond the cell contour
                    ribs = []
                    for riblong,riblongplot in zip(riblongs,riblongplots):
                        inter = riblong.GetIntersect(boundshape)
                        riblongplot.pop(0).remove()
                        ribs.append(shape.Shape(inter))
                        ribx = (inter[0][0], inter[1][0])
                        riby = (inter[0][1], inter[1][1])
                        ribplots.append(ax1.plot(ribx, riby, c='y'))
                    ax1.imshow(background + 1000*skeletonize(msnake.levelset), cmap=ppl.cm.gray)
                    ax_u.set_data(msnake.levelset)
                    fig.canvas.draw()
                    
                #code to make/fit/show capsule
                ribLens = [rib.avgDist for rib in ribs]
                midRibLens = ribLens[int(len(ribs)*(3/10.0)):]    #:int(len(ribs)*(7/10.0))]
                avgRibLen = np.mean(midRibLens)
                rad = avgRibLen/2
                cens = []
                for i,j in ((1,0),(0,1)):
                    cenVec = rad*((inters[i]-inters[j])/np.sqrt(np.dot((inters[i]-inters[j]), (inters[i]-inters[j]))))
                    cens.append(inters[j] + cenVec)
                semiSlope = -(inters[1][0] - inters[0][0])/(inters[1][1] - inters[0][1])
                Xs,Ys = [],[]
                for i,cen in zip((1,-1),cens):
                    xlow = cen[0] - rad
                    xhigh = cen[0] + rad
                    xall = np.arange(xlow, xhigh, (xhigh-xlow)/500).tolist()
                    ytop = [cen[1] + np.sqrt(rad**2 - (x - cen[0])**2) for x in xall]
                    ybot = [cen[1] - np.sqrt(rad**2 - (x - cen[0])**2) for x in xall]
                    semiB = cen[1] - semiSlope*cen[0]
                    X,Y = [],[]
                    #the whole lastInSemi thing is to avoid ending up with closed semicircles at the ends of the capsule shape
                    startInSemi = (ybot[0] - semiSlope*xall[0] - semiB)/np.sqrt(semiSlope**2 + 1)*i > 0
                    lastInSemi = False
                    for x,y in zip(xall+xall[::-1], ybot+ytop[::-1]):
                        nowInSemi = (y - semiSlope*x - semiB)/np.sqrt(semiSlope**2 + 1)*i > 0
                        if nowInSemi:
                            if not lastInSemi:
                                X.append([])
                                Y.append([])
                            X[-1].append(x)
                            Y[-1].append(y)
                        lastInSemi = nowInSemi
                    if startInSemi:
                        X = list(itertools.chain(*X[::-1]))
                        Y = list(itertools.chain(*Y[::-1]))
                    else:
                        X = list(itertools.chain(*X))
                        Y = list(itertools.chain(*Y))
                    Xs.append(X)
                    Ys.append(Y)
                Xs,Ys = list(itertools.chain(*Xs)), list(itertools.chain(*Ys))
                #closing the capsule shape for demonstration/drawing purposes
                cXs = Xs + [Xs[0]]
                cYs = Ys + [Ys[0]]
                capsule = ax1.plot(cXs, cYs, c='g', linewidth=4)
            except UnboundLocalError:
                pass
        ax1.imshow(background + 1000*skeletonize(msnake.levelset), cmap=ppl.cm.gray)
        ax_u.set_data(msnake.levelset)
        fig.canvas.draw()
        
        #ppl.pause(0.001)
    
    #save final figure if filename is specified
    if outfile:
        ppl.savefig('%s%.4d' % (outfile, i))
    # Return the last levelset.
    return msnake.levelset

def evolve_visual3d(msnake, levelset=None, num_iters=20):
    """
    Visual evolution of a three-dimensional morphological snake.
    
    Parameters
    ----------
    msnake : MorphGAC or MorphACWE instance
        The morphological snake solver.
    levelset : array-like, optional
        If given, the levelset of the solver is initialized to this. If not
        given, the evolution will use the levelset already set in msnake.
    num_iters : int, optional
        The number of iterations.
    """
    from enthought.mayavi import mlab
    
    if levelset is not None:
        msnake.levelset = levelset
    
    fig = mlab.gcf()
    mlab.clf()
    src = mlab.pipeline.scalar_field(msnake.data)
    mlab.pipeline.image_plane_widget(src, plane_orientation='x_axes', colormap='gray')
    cnt = mlab.contour3d(msnake.levelset, contours=[0.5])
    
    for i in xrange(num_iters):
        msnake.step()
        cnt.mlab_source.scalars = msnake.levelset
    
    # Return the last levelset.
    return msnake.levelset

def circle_levelset(shape, center, sqradius, scalerow=1.0):
    """Build a binary function with a circle as the 0.5-levelset."""
    R, C = np.mgrid[:shape[0], :shape[1]]
    phi = sqradius - (np.sqrt(scalerow*(R-center[0])**2 + (C-center[1])**2))
    u = np.float_(phi>0)
    return u

def outline(im):
    ''' Input binary 2D (NxM) image. Ouput array (2xK) of K (y,x) coordinates
        where 0 <= K <= 2*M.
    '''
    topbottom = np.empty((1,2*im.shape[1]), dtype=np.uint16)
    topbottom[0,0:im.shape[1]] = np.argmax(im, axis=0)
    topbottom[0,im.shape[1]:] = (im.shape[0]-1)-np.argmax(np.flipud(im), axis=0)
    mask      = np.tile(np.any(im, axis=0), (2,))
    xvalues   = np.tile(np.arange(im.shape[1]), (1,2))
    return np.vstack([topbottom,xvalues])[:,mask].T