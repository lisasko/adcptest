from abc import ABC, abstractmethod
import numpy as np
from typing import Sequence, Optional, Union

from qrevint_21_03.Classes.TransformationMatrix import TransformationMatrix




"""
    Base class to obtain the instrument calibration matrix :
    This class can be subclassed to add a method for providing the Beam to Instrument calibration matrix.

    Methods:
        has_data(adcp): Returns whether the provider (or list of providers) has the required data.
        i2b_matrix(adcp): Returns the instrument-to-beam transformation matrix.
        b2i_matrix(adcp): Returns the beam-to-instrument transformation matrix.
        get_has_data(adcp): Abstract method to check if the provider has the required data.
        get_i2b_matrix(adcp): Abstract method to compute the instrument-to-beam matrix.
        get_b2i_matrix(adcp): Abstract method to compute the beam-to-instrument matrix.

"""


class InstrumentMatrixProvider(ABC):

    def has_data(self, adcp) -> Union[bool, np.array] :

        if isinstance(self, list):
            return np.array([p.get_has_data(adcp) for p in self], dtype=bool)
        return self.get_has_data(adcp)
    
    def i2b_matrix(self, adcp) -> np.ndarray:

        providers = [self] if not isinstance(self, list) else self

        for p in providers:
            if p.get_has_data(adcp):
                return p.get_i2b_matrix(adcp)

        raise RuntimeError("Cannot compute instrument-to-beam transformation matrix: no valid provider.")
    
    def b2i_matrix(self, adcp) -> np.ndarray:

        providers = [self] if not isinstance(self, list) else self

        for p in providers:
            if p.get_has_data(adcp):
                return p.get_b2i_matrix(adcp)

        raise RuntimeError("Cannot compute beam-to-instrument transformation matrix: no valid provider.")


    @abstractmethod
    def get_has_data(self, adcp) -> bool:
        pass

    @abstractmethod
    def get_i2b_matrix(self, adcp) -> np.ndarray: 
        pass
    @abstractmethod
    def get_b2i_matrix(self, adcp) -> np.ndarray: 
        pass


def _to_4x4stack(arr: np.ndarray) -> np.ndarray:

    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return np.tile(np.eye(4)[None, :, :], (1, 1, 1))[0:1, :, :]
    # common formats: (4,4), (4,4,n), (n,4,4), (1,n,4,4) etc.
    if a.ndim == 2 and a.shape == (4, 4):
        return a[None, :, :]
    if a.ndim == 3:
        # if shape is (4,4,n) move axis
        if a.shape[0] == 4 and a.shape[1] == 4:
            # ambiguous: could be (4,4,n) or (n,4,4). prefer (n,4,4) if first dim !=4
            if a.shape[2] == 4:
                return a
            # else assume (4,4,n)
            return np.moveaxis(a, 2, 0)
        if a.shape[-2:] == (4, 4):
            return a
    if a.ndim == 4:
        # (1, n, 4,4) -> reduce to (n,4,4)
        if a.shape[-2:] == (4, 4):
            # collapse leading dims into n
            n = int(np.prod(a.shape[:-2]))
            return a.reshape(n, 4, 4)
    raise ValueError("Cannot normalize transformation matrix shape: %s" % (a.shape,))



"""
    Computes uncalibrated instrument matrix from the beam angle.
    Provider for instrument-to-beam transformation matrices based on beam angle and convexity.

"""

class InstrumentMatrixFromBAngle(InstrumentMatrixProvider):

    def get_has_data(self, adcp) -> bool:

        ba = getattr(adcp, "beam_angle_deg", None)
        if ba is None:
            return False
        return np.all(np.isfinite(np.asarray(ba, dtype=float)))

    def get_i2b_matrix(self, adcp) -> np.ndarray:

        bangle = np.asarray(getattr(adcp, "beam_angle_deg", 20.0), dtype=float).reshape(-1)
        c = np.asarray(getattr(adcp, "convexity", 1.0), dtype=float)

        if c.shape == (): 
            c = np.full_like(bangle, float(c), dtype=float)
        else:
            c = np.asarray(c, dtype=float).reshape(-1)
            if c.size != bangle.size:
                c = np.broadcast_to(c.ravel()[0], bangle.shape)

        a = np.sin(np.deg2rad(bangle))
        b = np.cos(np.deg2rad(bangle))
        d = np.sqrt(2.0) * a / 2.0
        zr = np.zeros_like(a)

        n = max(1, bangle.size)
        mats = np.zeros((n, 4, 4), dtype=float)

        mats[:, 0, 0] = c * a
        mats[:, 0, 1] = zr
        mats[:, 0, 2] = b
        mats[:, 0, 3] = d

        mats[:, 1, 0] = -c * a
        mats[:, 1, 1] = zr
        mats[:, 1, 2] = b
        mats[:, 1, 3] = d

        mats[:, 2, 0] = zr
        mats[:, 2, 1] = -c * a
        mats[:, 2, 2] = b
        mats[:, 2, 3] = -d

        mats[:, 3, 0] = zr
        mats[:, 3, 1] = c * a
        mats[:, 3, 2] = b
        mats[:, 3, 3] = -d

        return mats[None, :, :, :]

    def get_b2i_matrix(self, adcp) -> np.ndarray:
        i2b = self.get_i2b_matrix(adcp)
        mats = _to_4x4stack(i2b)
        invs = np.zeros_like(mats)

        for k in range(mats.shape[0]):
            invs[k] = np.linalg.inv(mats[k])
        return invs[None, :, :, :]



"""
    Gets calibrated instrument matrix from pd0 files.
    Provider for instrument-to-beam transformation matrices from calibration data.

    Using the TransformationMatrix class from QRevInt code. 

"""

class InstrumentMatrixFromCalibration(InstrumentMatrixProvider):

    def get_has_data(self, adcp) -> bool:
        if hasattr(adcp, "t_matrix") and adcp.t_matrix is not None:
            try:
                mat = getattr(adcp.t_matrix, "matrix", None)
                return mat is not None
            except Exception:
                return False

        raw = getattr(adcp, "_raw", None)
        if isinstance(raw, dict) and "transformation_matrix" in raw:
            return True
        
        return False
    

    def get_b2i_matrix(self, adcp) -> np.ndarray:

        if hasattr(adcp, "t_matrix") and adcp.t_matrix is not None:
            mat = getattr(adcp.t_matrix, "matrix", None)
            if mat is not None:
                mats = _to_4x4stack(mat)
                return mats[None, :, :, :]

        raw = getattr(adcp, "_raw", None)
        if isinstance(raw, dict) and "transformation_matrix" in raw:
            mat = np.asarray(raw["transformation_matrix"]["matrix"], dtype=float)
            mats = _to_4x4stack(mat)
            return mats[None, :, :, :]
        
        raise RuntimeError("No calibration matrix available in adcp")
    

    def get_i2b_matrix(self, adcp) -> np.ndarray:

        b2i = self.get_b2i_matrix(adcp)
        mats = _to_4x4stack(b2i)
        invs = np.zeros_like(mats)
        for k in range(mats.shape[0]):
            invs[k] = np.linalg.inv(mats[k])
        return invs[None, :, :, :]



"""
    Gets calibrated instrument matrix from ps3 command output.

    Properties : 
        ps3_matrix - beam to instrument matrix from ps3 command.

"""

class InstrumentMatrixFromPS3(InstrumentMatrixProvider):

    def __init__(self, ps3_matrix: Optional[np.ndarray] = None):
        self.ps3_matrix = np.eye(4) if ps3_matrix is None else np.asarray(ps3_matrix, dtype=float)

    def get_has_data(self, adcp) -> bool:
        return not np.allclose(self.ps3_matrix, np.eye(4))

    def get_i2b_matrix(self, adcp) -> np.ndarray:
        n = max(1, int(getattr(adcp, "nensembles", getattr(adcp, "n_ensembles", 1))))
        return np.tile(self.ps3_matrix[None, :, :], (1, n, 1, 1))

    def get_b2i_matrix(self, adcp) -> np.ndarray:
        invm = np.linalg.inv(self.ps3_matrix)
        n = max(1, int(getattr(adcp, "nensembles", getattr(adcp, "n_ensembles", 1))))
        return np.tile(invm[None, :, :], (1, n, 1, 1))