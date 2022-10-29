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


def ECI_2_SB(ECI, quat):
    q0 = quat[0]
    q1 = quat[1]
    q2 = quat[2]
    q3 = quat[3]

    rot_mat = np.array(
        [
            [
                (q0**2 + q1**2 - q2**2 - q3**2),
                2 * (q1 * q2 - q0 * q3),
                2 * (q0 * q2 + q1 * q3),
            ],
            [
                2 * (q1 * q2 + q0 * q3),
                (q0**2 - q1**2 + q2**2 - q3**2),
                2 * (q2 * q3 - q0 * q1),
            ],
            [
                2 * (q1 * q3 - q0 * q2),
                2 * (q0 * q1 + q2 * q3),
                (q0**2 - q1**2 - q2**2 + q3**2),
            ],
        ]
    )
    rot_mat = np.transpose(rot_mat)
    SB = np.dot(rot_mat, ECI)

    return SB


def get_sunpos(t):
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
            sun_SB = ECI_2_SB(sun_ECI, quaternion)

        else:
            sun_SB = [0, 0, 0]
            sun_ECI = [0, 0, 0]

        return pos_GCRS, v_GCRS, lla, sun_SB, sun_ECI
