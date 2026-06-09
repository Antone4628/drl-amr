"""Mesh generation for Discontinuous Galerkin discretization.

This module provides grid generation for DG methods, creating node coordinates
and connectivity arrays from element boundary positions.

Key Functions:
    create_grid_us: Generate unstructured DG grid with variable element sizes.

Note:
    Grids are generated using LGL node distribution within each element.
    The 'us' suffix indicates unstructured (variable element size) support.
"""
import numpy as np


def create_grid_us(ngl: int, nelem: int, npoin_cg: int, npoin_dg: int, xgl, xelem):
    """Create DG computational grid with variable element sizes.
    
    Generates node coordinates and connectivity arrays for discontinuous
    Galerkin discretization. Supports unstructured grids where element
    sizes can vary (e.g., after AMR adaptation).
    
    Args:
        ngl: Number of LGL points per element.
        nelem: Number of elements in the grid.
        npoin_cg: Number of points in continuous Galerkin grid.
        npoin_dg: Number of points in discontinuous Galerkin grid.
        xgl: LGL node positions in reference element [-1, 1], shape (ngl,).
        xelem: Element boundary positions, shape (nelem + 1,).
        
    Returns:
        coord_dg: Physical coordinates of DG nodes, shape (npoin_dg,).
        intma_dg: Connectivity matrix, shape (ngl, nelem). Maps local node
            index to global node number: intma_dg[i, e] = global index.
        periodicity_dg: Periodicity pointer array, shape (npoin_dg,).
    
    Note:
        The function internally generates both CG and DG grids but only
        returns DG data. Grid points are distributed according to LGL
        distribution within each element, mapped from [-1, 1] to physical
        coordinates defined by xelem.
    """

    #Initialize
    npin_cg = int(npoin_cg)
    npin_dg = int(npoin_dg)
    intma_dg=np.zeros([ngl,nelem], dtype = int)
    intma_cg=np.zeros([ngl,nelem], dtype = int)
    periodicity_cg = np.zeros(npin_cg, dtype = int)
    periodicity_dg = np.zeros(npin_dg, dtype = int)


    #Constants
    # xmin = -1
    xmin = xelem[0]
    # xmax = 1
#     dx = (xmax-xmin)/nelem
    coord_cg = np.zeros(npoin_cg)
    coord_dg = np.zeros(npoin_dg)


    #generate COORD and INTMA for CG
    ip=0
    dx = 0
    x0 = xmin
    coord_cg[0]=xmin
    for e in range(nelem): #0,1,2,3
        x0 = x0 + dx
        dx = xelem[e+1]-xelem[e]
#         print(f'element: {e}, dx: {dx}')

#         print(f'x0: {x0}')
#         intma_cg[0][e] = ip+1 #produces same as MATLAB
        intma_cg[0][e] = ip
        for i in range(1, ngl): #1,2,3
#             print(f'e = {e}, i={i}')
            ip+=1
            # print(f'ip={ip}, i = {i}')

            coord_cg[ip]=(xgl[i]+1)*(dx/2) + x0

#             print(coord_cg[ip])

#             intma_cg[i][e]=ip+1 #produces same as MATLAB
            intma_cg[i][e]=ip
#     print(f'cg coords:\n {coord_cg}')
#     print(f'intma_cg:\n{intma_cg}')

    #Generate periodicity pointer for CG
    for i in range(npoin_cg):
        periodicity_cg[i]=i
    periodicity_cg[-1] = periodicity_cg[0] # maybe use -1




    #generate COORD and INTMA for DG
    ip=0
    for e in range(nelem):
        for i in range(ngl):
#             ip+=1
            intma_dg[i][e] = ip
            ip+=1
#     print(f'intma_dg:\n {intma_dg}')
    for e in range(nelem):
        for i in range(ngl):
#             print(f'e = {e}, i={i}')
            ip_cg = intma_cg[i][e]
            ip_dg = intma_dg[i][e]
            coord_dg[int(ip_dg)] = coord_cg[int(ip_cg)]

    for i in range(npoin_dg):
        periodicity_dg[i]=i



    return  coord_dg, intma_dg, periodicity_dg


