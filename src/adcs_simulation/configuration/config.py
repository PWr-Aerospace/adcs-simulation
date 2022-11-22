"""Module that handles configuration of simulation."""

import json
import os
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).parents[3]
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DATA_DIR = os.path.join(ROOT_DIR, "example_data")


class Config:
    """
    Get the configuration data from setup.
    """

    def __init__(self, setup_file: str) -> None:
        setup_path = os.path.join(CONFIG_DIR, setup_file)
        self._setup = self._read_setup(setup_path)

    def _read_setup(self, setup_file: str) -> Any:
        """
        Read the setup file.

        Parameters
        ----------
        setup_file : str - name of the file. Default setup.json

        Returns
        -------
        Loaded json file.

        Raises
        ------
        RuntimeError -  raise error if the setup file was not found.
        """
        if not Path(setup_file).exists():
            raise RuntimeError(f"Setup file not found at: {setup_file}")
        with open(Path(setup_file), "r") as file:
            return json.load(file)

    @property
    def euler_angles(self) -> tuple:
        """
        Get the initial Euler Angles.

        Returns
        -------
        phi0 : int - initial Euler Angle around the X Axis in degrees
        theta0 : int - initial Euler Angle around the Y Axis in degrees
        psi0 : int - initial Euler Angle around the Z Axis in degrees
        """
        euler_angles = self._setup["Initial"][0]["EulerAngles"]
        phi0 = euler_angles[0]
        theta0 = euler_angles[1]
        psi0 = euler_angles[2]

        return phi0, theta0, psi0

    @property
    def rotation(self) -> tuple:
        """
        Get the initial rotation velocity.

        Returns
        -------
        p0 : float - initial rotation velocity around the X Axis in rad/s
        q0 : float - initial rotation velocity around the Y Axis in rad/s
        r0 : float - initial rotation velocity around the Z Axis in rad/s
        """
        omega = self._setup["Initial"][0]["rotVelocity"]
        p0 = omega[0]
        q0 = omega[1]
        r0 = omega[2]

        return p0, q0, r0

    @property
    def iterations_info(self) -> tuple:
        """
        Get the time control settings.

        Returns
        -------
        time0 : int - beginning time, default 0
        tend : int - end time in seconds
        tstep : int - time step, default 1 second
        """
        time0 = self._setup["iterations"][0]["start"]
        tend = self._setup["iterations"][0]["stop"]
        tstep = self._setup["iterations"][0]["step"]

        return time0, tend, tstep

    @property
    def torquer_params(self) -> tuple:
        """
        Get the parameters of the magnetorquer.

        Returns
        -------
        n : int - number of coils on one solenoid
        area : int/float - solenoid cross-section area
        """

        n = self._setup["magnetorquer"][0]["n"]
        area = self._setup["magnetorquer"][0]["A"]

        return n, area

    @property
    def sat_params(self) -> tuple:
        """
        Satellite kinematic parameters

        Returns
        -------
        inertia : list - inertia matrix of the mass
        mass : int/float - satellite mass
        """

        inertia = self._setup["Satelite"][0]["I"]
        mass = self._setup["Satelite"][0]["mass"]

        return inertia, mass

    @property
    def planet_data(self) -> tuple:
        """
        Earth constants

        Returns
        -------
        gravity_const : float - gravitational constant
        earth_mass : float - Earth mass in kg
        earth_radius : float - Earth radius in m
        """

        gravity_const = self._setup["PlanetConst"][0]["G"]
        earth_mass = self._setup["PlanetConst"][0]["M"]
        earth_radius = self._setup["PlanetConst"][0]["R"]

        return gravity_const, earth_mass, earth_radius
