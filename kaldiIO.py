#!/usr/bin/python3

## Author: D S Pavan Kumar
## Email: Pavan.Kumar@inin.com

import numpy
import struct

## Read utterance
def readUtterance (ark):
    ## Read utterance ID
    uttId = b''
    c = ark.read(1)
    if not c:
        return None, None
    while c != b' ':
        uttId += c
        c = ark.read(1)
    ## Read feature matrix
    header = struct.unpack('<xcccc', ark.read(5))
    m, rows = struct.unpack('<bi', ark.read(5))
    n, cols = struct.unpack('<bi', ark.read(5))
    featMat = numpy.frombuffer(ark.read(rows * cols * 4), dtype=numpy.float32)
    return uttId.decode(), featMat.reshape((rows,cols))

def writeUtterance (uttId, featMat, ark, encoding):
    featMat = numpy.asarray (featMat, dtype=numpy.float32)
    m,n = featMat.shape
    ## Write header
    ark.write (struct.pack('<%ds'%(len(uttId)),uttId.encode(encoding)))
    ark.write (struct.pack('<cxcccc',' '.encode(encoding),'B'.encode(encoding),
                'F'.encode(encoding),'M'.encode(encoding),' '.encode(encoding)))
    ark.write (struct.pack('<bi', 4, m))
    ark.write (struct.pack('<bi', 4, n))
    ## Write feature matrix
    ark.write (featMat)

