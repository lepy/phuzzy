LS-DYNA
=======


.. code-block:: lsdyna

    *AIRBAG_ALE
    $#     sid    sidtyp                                               mwd      spsf
             0         0                                               0.0       0.0
    $#  atmost    atmosp                  gc        cc      tvol   tfpress
    &param_1         0.0                 0.0&param_2         0.0       0.0
    $#   nquad     ctype      pfac      fric    frcmin   normtyp     ileak     pleak
             4         4       0.1       0.0       0.3         0         2       0.1
    $#   ivset    ivtype    iblock      vcof
             0         0         0       0.0
    $#nx_idair  ny_idgas        nz    movern      zoom
             0         0         1         0         0
    $#      x0        y0        z0        x1        y1        z1
           0.0       0.0       0.0       0.0       0.0       0.0
    $#      x2        y2        z2        x3        y3        z3
           0.0       0.0       0.0       0.0       0.0       0.0
    $#  swtime    unused        hg      nair      ngas     norif     lcvel       lct
           0.0                 0.0         0         0         0         0         0

.. code-block:: none

    *AIRBAG_ALE
    $#     sid    sidtyp                                               mwd      spsf
             0         0                                               0.0       0.0
    $#  atmost    atmosp                  gc        cc      tvol   tfpress
    &param           0.0                 0.0       1.0       0.0       0.0
    $#   nquad     ctype      pfac      fric    frcmin   normtyp     ileak     pleak
             4         4       0.1       0.0       0.3         0         2       0.1
    $#   ivset    ivtype    iblock      vcof
             0         0         0       0.0
    $#nx_idair  ny_idgas        nz    movern      zoom
             0         0         1         0         0
    $#      x0        y0        z0        x1        y1        z1
           0.0       0.0       0.0       0.0       0.0       0.0
    $#      x2        y2        z2        x3        y3        z3
           0.0       0.0       0.0       0.0       0.0       0.0
    $#  swtime    unused        hg      nair      ngas     norif     lcvel       lct
           0.0                 0.0         0         0         0         0         0
