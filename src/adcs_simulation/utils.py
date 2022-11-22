"""
Module with utility functions.

"""


from dataclasses import dataclass


@dataclass
class TLE:
    """
    Class dedicated to read given two-line-element (TLE)
    """

    first_line: str
    second_line: str

    sat_number: str
    classification: str
    international_designator: str
    year: str
    day: str
    epoch: str
    first_mean_motion: str
    second_mean_motion: str
    second_mean_motion_exp: str
    bstar_drag: str
    bstar_drag_exp: str
    ephemeris: str
    norad_number: str

    inclination: str
    right_ascension: str
    eccentricity: str
    arg_perigee: str
    mean_anomaly: str
    mean_motion: str
    revolutions: str


def _convert_tle_year(tle: TLE) -> None:
    """
    Check the year written in TLE file.

    Parameters
    ----------
    tle : TLE - two-line-element for the satellite

    """

    if 57 <= tle.year <= 99:
        tle.year += 1900
    elif 0 <= tle.year <= 56:
        tle.year += 2000
    else:
        raise ValueError(f"Invalid year in TLE data: {tle.year=}")


def read_tle(tle_file: str) -> TLE:
    """
    Read the TLE from txt file.

    Parameters
    ----------
    tle_file : str - TLE file name

    Returns
    -------
    TLE
    """

    with open(tle_file) as file:
        tle = file.read()
    tle_lines = tle.split(sep="\n")
    tle = tle.split(sep=" ")

    tle[:] = [x for x in tle if x]
    _tle = TLE(
        first_line=tle_lines[0],
        second_line=tle_lines[1],
        sat_number=tle[1][:-1],
        classification=tle[1][-1],
        international_designator=tle[2],
        year=int(tle[3][0:2]),
        day=tle[3][2:5],
        epoch="0{tle[3][5:]}",
        first_mean_motion=tle[4],
        second_mean_motion=tle[5][:-2],
        second_mean_motion_exp=tle[5][-2:],
        bstar_drag=tle[6][:-2],
        bstar_drag_exp=tle[6][-2:],
        ephemeris=tle[7],
        norad_number=tle[8][:-2],
        inclination=tle[10],
        right_ascension=tle[11],
        eccentricity=tle[12],
        arg_perigee=tle[13],
        mean_anomaly=tle[14],
        mean_motion=tle[15][:-5],
        revolutions=tle[15][-5:],
    )
    _convert_tle_year(_tle)
    return _tle
