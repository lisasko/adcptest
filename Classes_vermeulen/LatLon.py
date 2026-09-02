from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Sequence
import utm
from abc import ABC, abstractmethod
from pyproj import Transformer

from .ADCPHorizontalPosition import ADCPHorizontalPosition


## Classe LatLonProvider issue de LatLonProvider.m ##
"""
Classe mère permettant d'optenir les coordonnées géographiques à partir des données ADCP.
(latitude et longitude)
"""

class LatLonProvider:

    def has_data(self, adcp) -> bool: # returns whether geographic data is available
        return self.get_has_data(adcp)
    
    def get_has_data(self, adcp) -> bool:  
        raise NotImplementedError
    
    def lat_lon(self, adcp) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_lat_lon(adcp)
    
    def get_lat_lon(self, adcp) -> Tuple[np.ndarray, np.ndarray]:  
        raise NotImplementedError
    

## Classe LatLonTfiles issue de LatLonTfiles.m ##
"""
Lit les coordonnées géographiques depusi les fichiers de transect.
"""

class LatLonTfiles(LatLonProvider):

    def get_has_data(self, adcp) -> bool:
        raw = getattr(adcp, "_raw", None)
        return isinstance(raw, dict) and "tFiles" in raw and all(k in raw["tFiles"] for k in ("lat", "long"))
    
    def get_lat_lon(self, adcp):
        raw = adcp._raw
        lat = np.asarray(raw["tFiles"]["lat"], dtype=float).reshape(1, -1)[0]
        lon = np.asarray(raw["tFiles"]["long"], dtype=float).reshape(1, -1)[0]
        return lat, lon


## Classe LatLonVisea issue de LatLonVisea.m ##
"""
Lit les coordonnées géographiques depuis les fichiers externes VISEA.
"""

class LatLonVisea(LatLonProvider):

    def get_has_data(self, adcp) -> bool:
        raw = getattr(adcp, "_raw", None)
        return isinstance(raw, dict) and "VISEA_Extern" in raw and all(k in raw["VISEA_Extern"] for k in ("Latitudeseconds", "Longitudeseconds"))
    
    def get_lat_lon(self, adcp):
        raw = adcp._raw["VISEA_Extern"]
        lat = np.asarray(raw["Latitudeseconds"], dtype=float) / 3600.0
        lon = np.asarray(raw["Longitudeseconds"], dtype=float) / 3600.0
        return lat.reshape(-1), lon.reshape(-1)


## Classe LatLonGGA issue de LatLonGGA.m ##
"""
Lit les coordonnées géographiques depuis les données GGA (Global Positioning System Fix Data) présentes dans les données brutes de l'ADCP.
"""

class LatLonGGA(LatLonProvider):

    def get_has_data(self, adcp) -> bool:
        raw = getattr(adcp, "_raw", None)
        return isinstance(raw, dict) and "GGA" in raw and all(k in raw["GGA"] for k in ("lat", "long"))
    
    def get_lat_lon(self, adcp):
        raw = adcp._raw["GGA"]
        lat = np.asarray(raw["lat"], dtype=float).reshape(-1)
        lon = np.asarray(raw["long"], dtype=float).reshape(-1)
        return lat, lon


## Classe LatLonNfilesGGA issue de LatLonNfilesGGA.m ##
"""
Lit les coordonnées géographiques depuis les string GGA  présent dans les n-files.
"""

class LatLonNfilesGGA(LatLonProvider):

    def get_has_data(self, adcp) -> bool:
        raw = getattr(adcp, "_raw", None)
        return isinstance(raw, dict) and "nFiles" in raw and "GGA" in raw["nFiles"] and all(k in raw["nFiles"]["GGA"] for k in ("lat", "long"))
    
    def get_lat_lon(self, adcp):
        raw = adcp._raw["nFiles"]["GGA"]
        lat = np.asarray(raw["lat"], dtype=float).reshape(-1)
        lon = np.asarray(raw["long"], dtype=float).reshape(-1)
        return lat, lon


## Classe LatLonNMEAGGA issue de LatLonNMEAGGA.m ##
"""
Lit les coordonnées géographiques depuis les données NMEA GGA.
"""

class LatLonNMEAGGA(LatLonProvider):
    
    def get_has_data(self, adcp) -> bool:
        raw = getattr(adcp, "_raw", None)
        return isinstance(raw, dict) and "NMEAGGA" in raw and "Lat" in raw["NMEAGGA"] and "Long" in raw["NMEAGGA"]
    
    def get_lat_lon(self, adcp):
        raw = adcp._raw["NMEAGGA"]
        lat_arr = np.asarray(raw["Lat"], dtype=float)
        lon_arr = np.asarray(raw["Long"], dtype=float)
        dt = np.asarray(raw.get("deltaT", np.zeros_like(lat_arr)), dtype=float)

        # Robust selection: if deltaT is 2D and matches lat shape, pick closest per ensemble
        if dt.ndim == 2 and lat_arr.ndim == 2:
            # dt shape (n_samples, n_ens), lat_arr shape same => choose per ensemble
            idx = np.argmin(np.abs(dt), axis=0)
            n_ens = lat_arr.shape[1]
            chosen_lat = lat_arr[idx, np.arange(n_ens)]
            chosen_lon = lon_arr[idx, np.arange(n_ens)]
        else:
            # fallback: flatten and return first valid series
            chosen_lat = lat_arr.reshape(-1)
            chosen_lon = lon_arr.reshape(-1)
        # sanitize zeros -> NaN
        mask = (chosen_lat == 0) & (chosen_lon == 0)
        chosen_lat = np.where(mask, np.nan, chosen_lat)
        chosen_lon = np.where(mask, np.nan, chosen_lon)
        return np.asarray(chosen_lat, dtype=float), np.asarray(chosen_lon, dtype=float)


## Classe LatLonToProjection issue de LatLonToProjection.m ##
"""
Classe de base permettant de convertir les coordonnées géographiques (latitude et longitude) 
en coordonnées projetées (x, y) dans un système de projection défini.
"""

class LatLonToProjection(ADCPHorizontalPosition):

    def __init__(self, ll_provider: Optional[Sequence[LatLonProvider]] = None):
        if ll_provider is None:
            ll_provider = [ProjectedCoordinatesFromViseaExtern(), LatLonVisea(), LatLonNfilesGGA(), LatLonTfiles(), LatLonNMEAGGA(), LatLonGGA()]
        self.ll_provider = list(ll_provider)

    # def has_data(self, adcp):
    #     return any(p.get_has_data(adcp) for p in self.ll_provider)
    
    def has_data(self, adcp):
        for p in self.ll_provider:
            if hasattr(p, "get_has_data") and callable(p.get_has_data):
                if p.get_has_data(adcp):
                    return True
            elif hasattr(p, "has_data") and callable(p.has_data):
                if p.has_data(adcp):
                    return True
        return False


    # def get_horizontal_position(self, adcp):
    #     # pick first provider that has data
    #     for p in self.ll_provider:
    #         if p.get_has_data(adcp):
    #             lat, lon = p.get_lat_lon(adcp)
    #             x, y = self.xy(lat, lon)
    #             return np.vstack((np.asarray(x), np.asarray(y)))
    #     # fallback: empty
    #     return np.empty((2, 0), dtype=float)
    
    def get_horizontal_position(self, adcp):
        # Support both LatLonProvider (lat/lon -> projection)
        # and direct projected providers (x/y already available).
        for p in self.ll_provider:
            if hasattr(p, "get_has_data") and callable(p.get_has_data):
                if p.get_has_data(adcp):
                    lat, lon = p.get_lat_lon(adcp)
                    x, y = self.xy(lat, lon)
                    return np.vstack((np.asarray(x), np.asarray(y)))

            elif hasattr(p, "has_data") and callable(p.has_data):
                if p.has_data(adcp):
                    xy = p.get_horizontal_position(adcp)
                    return np.asarray(xy, dtype=float)

        return np.empty((2, 0), dtype=float)


    def xy(self, lat, lon):
        raise NotImplementedError

## Classe LatLonToUTM issue de LatLonToUTM.m ##
"""
Classe de projection dans le système de coordonnées UTM (Universal Transverse Mercator).
"""

class LatLonToUTM(LatLonToProjection):

    description = "UTM"

    def __init__(self, zone: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.zone = zone
    

    def xy(self, lat, lon):
        lat = np.asarray(lat, dtype=float).reshape(-1)
        lon = np.asarray(lon, dtype=float).reshape(-1)
        xs = np.empty_like(lat, dtype=float)
        ys = np.empty_like(lat, dtype=float)

        transformer = Transformer.from_crs(
            "+proj=longlat +ellps=intl +towgs84=-87,-98,-121,0,0,0,0 +no_defs",  # ED50 (lat/lon)
            "+proj=utm +zone=31 +ellps=intl +towgs84=-87,-98,-121,0,0,0,0 +units=m +no_defs",  # ED50 / UTM 31N
        )

        # transformer = Transformer.from_crs(
        #     "EPSG:4326",
        #     f"+proj=utm +zone=31 +datum=WGS84 +units=m +no_defs",
        #     always_xy=True,
        # )
        
    
        # zone = self.zone
        # if zone is None:
        #     finite_lon = lon[np.isfinite(lon)]
        #     mean_lon = np.nanmean(finite_lon) if finite_lon.size > 0 else 0.0
        #     zone = int(np.floor((mean_lon + 180.0) / 6.0) + 1)
        #     self.zone = zone

        # transformer = Transformer.from_crs(
        #     "EPSG:4326", 
        #     f"+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs",
        #     always_xy=True,
        # )
        # ##

        for i, (la, lo) in enumerate(zip(lat, lon)):
            if np.isnan(la) or np.isnan(lo):
                xs[i] = np.nan
                ys[i] = np.nan
                continue
            e, n = transformer.transform(lo, la)
            xs[i] = e
            ys[i] = n
        return xs, ys
    

## Classe LatLonToRD issue de LatLonToRD.m ##
"""
Projection dans le système de coordonnées RD (Rijksdriehoeksmeting), utilisé principalement aux Pays-Bas.
"""

class LatLonToRD(LatLonToProjection):
    description = "RD"
    X0 = 155000.0
    Y0 = 463000.0
    phi0 = 52.15517440
    lam0 = 5.38720621

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def ll(self, x, y):
        Kp = np.array([0,2,0,2,0,2,1,4,2,4,1], dtype=float)
        Kq = np.array([1,0,2,1,3,2,0,0,3,1,1], dtype=float)
        Kpq = np.array([3235.65389,-32.58297,-0.24750,-0.84978,-0.06550,
                        -0.01709,-0.00738,0.00530,-0.00039,0.00033,-0.00012], dtype=float)

        dX = 1e-5 * (np.asarray(x, dtype=float) - self.X0)
        dY = 1e-5 * (np.asarray(y, dtype=float) - self.Y0)

        # ensure same shape for vectorized ops
        dX = np.asarray(dX)
        dY = np.asarray(dY)
        lat = np.zeros_like(dX, dtype=float)
        lon = np.zeros_like(dX, dtype=float)

        for kp, kq, kpq in zip(Kp, Kq, Kpq):
            lat = lat + kpq * (dX ** kp) * (dY ** kq)
        lat = self.phi0 + lat / 3600.0

        Lp = np.array([1,1,1,3,1,3,0,3,1,0,2,5], dtype=float)
        Lq = np.array([0,1,2,0,3,1,1,2,4,2,0,0], dtype=float)
        Lpq = np.array([5260.52916,105.94684,2.45656,-0.81885,0.05594,
                        -0.05607,0.01199,-0.00256,0.00128,0.00022,-0.00022,0.00026], dtype=float)

        for lp, lq, lpq in zip(Lp, Lq, Lpq):
            lon = lon + lpq * (dX ** lp) * (dY ** lq)
        lon = self.lam0 + lon / 3600.0

        return np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)

    def xy(self, lat, lon):
        Rp = np.array([0,1,2,0,1,3,1,0,2], dtype=float)
        Rq = np.array([1,1,1,3,0,1,3,2,3], dtype=float)
        Rpq = np.array([190094.945,-11832.228,-114.221,-32.391,-0.705,
                        -2.340,-0.608,-0.008,0.148], dtype=float)

        Sp = np.array([1,0,2,1,3,0,2,1,0,1], dtype=float)
        Sq = np.array([0,2,0,2,0,1,2,1,4,4], dtype=float)
        Spq = np.array([309056.544,3638.893,73.077,-157.984,59.788,
                        0.433,-6.439,-0.032,0.092,-0.054], dtype=float)

        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        dPhi = 0.36 * (lat - self.phi0)
        dLam = 0.36 * (lon - self.lam0)

        # broadcast to 1D arrays
        dPhi = np.asarray(dPhi).reshape(-1)
        dLam = np.asarray(dLam).reshape(-1)

        x = np.zeros_like(dPhi, dtype=float)
        y = np.zeros_like(dPhi, dtype=float)

        for rp, rq, rpq in zip(Rp, Rq, Rpq):
            x = x + rpq * (dPhi ** rp) * (dLam ** rq)
        x = self.X0 + x

        for sp, sq, spq in zip(Sp, Sq, Spq):
            y = y + spq * (dPhi ** sp) * (dLam ** sq)
        y = self.Y0 + y

        return x, y
    

# Classe ProjectedCoordinatesFromViseaExtern issue de ProjectedCoordinatesFromViseaExtern.m ##

# class ProjectedCoordinatesFromViseaExtern(ADCPHorizontalPosition):
#     description = ""

    # def has_data(self, adcp):

    #     try : 
    #         raw = getattr(adcp, "_raw", None)
            
    #         if raw is None :
    #             return False
            
    #         if not isinstance(raw, dict) or "VISEA_Extern" not in raw:
    #             return False
            
    #         visea_extern = raw["VISEA_Extern"]
            
    #         if not isinstance(visea_extern, dict):
    #             return False
            
    #         return all(k in visea_extern for k in ("Northing", "Easting"))
        
    #     except Exception : 
    #         return False
        
    # def get_horizontal_position(self, adcp):
    #     raw = adcp._raw["VISEA_Extern"]
    #     x = np.asarray(raw["Easting"], dtype=float)
    #     y = np.asarray(raw["Northing"], dtype=float)
    #     return np.vstack((x, y))


class ProjectedCoordinatesFromViseaExtern(ADCPHorizontalPosition):
    description = ""

    def has_data(self, adcp):

        if not hasattr(adcp, "transects"):
            return False
        
        for transect in adcp.transects:
            if hasattr(transect, "boat_vel") and hasattr(transect.boat_vel, "selected"):
                selected_vel = getattr(transect.boat_vel, transect.boat_vel.selected)
                if hasattr(selected_vel, "u_processed_mps") and hasattr(selected_vel, "v_processed_mps"):
                    return True
                
        return False

    def get_horizontal_position(self, adcp):

        all_x = []
        all_y = []

        for transect in adcp.transects:
            if hasattr(transect, "boat_vel") and hasattr(transect.boat_vel, "selected"):
                selected_vel = getattr(transect.boat_vel, transect.boat_vel.selected)
                if hasattr(selected_vel, "u_processed_mps") and hasattr(selected_vel, "v_processed_mps"):
                    track_x = np.nancumsum(selected_vel.u_processed_mps * transect.date_time.ens_duration_sec)
                    track_y = np.nancumsum(selected_vel.v_processed_mps * transect.date_time.ens_duration_sec)
                    all_x.append(track_x)
                    all_y.append(track_y)

        if not all_x:
            return np.vstack((np.array([]), np.array([])))
        
        return np.vstack((np.concatenate(all_x), np.concatenate(all_y)))