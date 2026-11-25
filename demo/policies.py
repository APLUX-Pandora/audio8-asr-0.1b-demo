"""Business-layer audio IO and independent offline segment boundaries."""
import math
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

def load_audio(path):
    pcm,sr=sf.read(path,dtype="float32",always_2d=True)
    pcm=pcm.mean(axis=1,dtype=np.float32)
    if sr!=16000:
        divisor=math.gcd(sr,16000)
        pcm=resample_poly(pcm,16000//divisor,sr//divisor).astype(np.float32)
    if not np.isfinite(pcm).all():raise ValueError("Audio contains non-finite samples")
    return pcm

def segments(length,seconds=25.0):
    size=int(seconds*16000)
    if size<201 or size>479800:raise ValueError("Segment duration must be in [0.013, 29.9875] seconds")
    if length<201:raise ValueError("Audio must contain more than 200 samples")
    cuts=[(i,min(i+size,length)) for i in range(0,length,size)]
    if len(cuts)>1 and cuts[-1][1]-cuts[-1][0]<201:
        cuts[-2:]=[(cuts[-2][0],cuts[-1][1])]
    return cuts
