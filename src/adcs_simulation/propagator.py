"""
Module for propagator calculations.

"""

import os
from dataclasses import dataclass

import numpy as np
import sgp4.api as sgp
import skyfield.api as skyfield
from scipy.spatial.transform import Rotation

from adcs_simulation.configuration.config import EXAMPLE_DATA_DIR
from adcs_simulation.utils import read_tle


def eci_2_sb_transformation(eci, quat):
    quat0 = quat[0]
    quat1 = quat[1]
    quat2 = quat[2]
    quat3 = quat[3]

    rot_mat = np.array(
        [
            [
                (quat0**2 + quat1**2 - quat2**2 - quat3**2),
                2 * (quat1 * quat2 - quat0 * quat3),
                2 * (quat0 * quat2 + quat1 * quat3),
            ],
            [
                2 * (quat1 * quat2 + quat0 * quat3),
                (quat0**2 - quat1**2 + quat2**2 - quat3**2),
                2 * (quat2 * quat3 - quat0 * quat1),
            ],
            [
                2 * (quat1 * quat3 - quat0 * quat2),
                2 * (quat0 * quat1 + quat2 * quat3),
                (quat0**2 - quat1**2 - quat2**2 + quat3**2),
            ],
        ]
    )
    rot_mat = np.transpose(rot_mat)
    sb = np.dot(rot_mat, eci)

    return sb


def get_sunpos(t: skyfield.timelib.Timescale):
    """
    Calculates the position of the Sun relative to Earth in the
    Earth Centered Inertial Frame.

    Parameters
    ----------
    t - time given as Julian Date in form of skyfield Timescale object

    Returns
    -------
    sunpos - list containing of 6 elements:
                sunpos[0] - X coordinate of the Sun vector in ECI frame
                sunpos[1] - Y coordinate of the Sun vector in ECI frame
                sunpos[2] - Z coordinate of the Sun vector in ECI frame
                sunpos[3] - right ascension -  angular distance of a
                particular point measured eastward along the celestial
                equator from the Sun at the March equinox, in radians
                sunspos[4] - 

    """
    t = t.tt
    t = t - 2451595
    rad = np.pi / 180
    mean_longtitude = np.mod((280.460 + 0.9856474 * t) * rad, 2 * np.pi)

    if mean_longtitude < 0:
        mean_longtitude = mean_longtitude + 2 * np.pi

    mean_anomaly = np.mod((357.528 + 0.9856003 * t) * rad, 2 * np.pi)
    if mean_anomaly < 0:
        mean_anomaly = mean_anomaly + 2 * np.pi

    mean_anomaly_2 = mean_anomaly + mean_anomaly

    if mean_anomaly_2 > 2 * np.pi:
        mean_anomaly_2 = np.mod(mean_anomaly_2, 2 * np.pi)

    ecliptic_longtitude = (
        mean_longtitude
        + (1.915 * np.sin(mean_anomaly) + 0.02 * np.sin(mean_anomaly_2)) * rad
    )
    sin_ecli_lo = np.sin(ecliptic_longtitude)
    cos_ecli_lo = np.cos(ecliptic_longtitude)

    # ecliptic_latitude = 0

    obliquity = (
        23.439 - 4.0e-7 * t
    ) * rad  # obliquity of the ecliptic (nachylenie osi)
    sin_obl_ecli = np.sin(obliquity)
    cos_obl_ecli = np.cos(obliquity)

    ASTRONOMICAL_UNIT = 149.60e09
    sunpos = [0, 0, 0, 0, 0, 0]
    sunpos[3] = np.arctan2(
        cos_obl_ecli * sin_ecli_lo, cos_ecli_lo
    )  # right ascension -  angular distance of a particular point measured eastward
    # along the celestial equator from the Sun at the March equinox, in radians
    if sunpos[3] < 0:
        sunpos[3] = sunpos[3] + 2 * np.pi

    sunpos[4] = np.arcsin(sin_obl_ecli * sin_ecli_lo)  # declination, in radians
    sunpos[5] = (
        1.00014 - 0.01671 * np.cos(mean_anomaly) - 1.4e-4 * np.cos(mean_anomaly_2)
    ) * ASTRONOMICAL_UNIT  # distance vector in meters

    # sun position in the the rectangular equatorial coordinate system in meteres
    sunpos[0] = sunpos[5] * cos_ecli_lo
    sunpos[1] = sunpos[5] * cos_obl_ecli * sin_ecli_lo
    sunpos[2] = sunpos[5] * sin_obl_ecli * sin_ecli_lo

    return sunpos


class Propagator:
    def __init__(self, tle_file: str) -> None:
        self._tle = read_tle(os.path.join(EXAMPLE_DATA_DIR, tle_file))

    def propagate(self, part_of_day, quaternion):
        """
        Performs calculations to obtain satellite position using SGP4 propagator

        Using the previously loaded TLE text file initializes the satellite model
        using skyfield library. The time given in form of Julian Date allows to
        propagate the object position on orbit in the GCRS frame (Geocentric
        Celestial Reference System - a type of Earth centered inertial coordinate
        frame which is related to its center of mass) as well as its kinematic
        parameters.
        A transformation to LLA format (Latitude, Longitude, Attitude) is also
        performed and the Sun vector (a vector pointing towards the Sun center)
        computed.

        Parameters
        ----------
        part_of_day - time from t0 given as 'part of day' to easy sum with JD
        quaternion - a 4-element array describing 3D rotations. The given
        quaternion gives the relation between the ECI frame (Earth Centered
        Inertial, in this case GCRS frame) and satellite orientation
        (Satellite Body Frame - SB). It can be used to obtain the rotation
        matrix or directly rotate vector between these reference frames.

        Returns
        -------
        pos_GCRS - object position in GCRS frame given in km.
        v_GCRS - velocity vector in GCRS frame given in km/s.
        lla - position given in LLA frame, attitude in km.
        sun_ECI - Sun vector given in the ECI frame. If the object is in
        Earths shadow returned as [0, 0, 0].
        sun_SB - Sun vector given in the SB frame. The transformation was
        obtained from sun_ECI using the given quaternion. If the object
        is in Earths shadow returned as [0, 0, 0].

        """
        satellite = skyfield.EarthSatellite(
            self._tle.first_line, self._tle.second_line
        )  # initialize TLE

        ts = skyfield.load.timescale()  # initialize time

        month, day, hour, minute, second = sgp.days2mdhms(
            self._tle.year, float(self._tle.day) + float(self._tle.epoch) + part_of_day
        )
        julian_date = ts.utc(self._tle.year, month, day, hour, minute, second)
        # TODO replace using skyfiled library

        # GCRS is and ECI frame almost similar do J2000
        GCRS = satellite.at(julian_date)
        pos_GCRS = GCRS.position.km
        v_GCRS = GCRS.velocity.km_per_s

        lat, lon = skyfield.wgs84.latlon_of(GCRS)

        # get the altitude as difference between Earth surface position and satelite position
        bluffton = skyfield.wgs84.latlon(lat.degrees, lon.degrees)
        position_difference = satellite - bluffton
        topocentric = position_difference.at(julian_date)
        alt, az, distance = topocentric.altaz()

        lla = [lat, lon, distance.km]

        eph = skyfield.load("de421.bsp")
        sunlight = satellite.at(julian_date).is_sunlit(eph)

        if sunlight:
            sunposition = get_sunpos(julian_date)
            sun_ECI = [sunposition[0], sunposition[1], sunposition[2]]
            sun_SB = eci_2_sb_transformation(sun_ECI, quaternion)

        else:
            sun_SB = [0, 0, 0]
            sun_ECI = [0, 0, 0]

        return pos_GCRS, v_GCRS, lla,sun_ECI, sun_SB
