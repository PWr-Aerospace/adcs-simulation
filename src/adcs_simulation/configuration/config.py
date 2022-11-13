"""Module that handles configuration of simulation."""

import json
import os
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).parents[3]
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DATA_DIR = os.path.join(ROOT_DIR, "example_data")


class Config:
    def __init__(self, setup_file: str) -> None:
        setup_path = os.path.join(CONFIG_DIR, setup_file)
        self._setup = self._read_setup(setup_path)

    def _read_setup(self, setup_file: str) -> Any:
        if not Path(setup_file).exists():
            raise RuntimeError(f"Setup file not found at: {setup_file}")
        with open(Path(setup_file), "r") as f:
            return json.load(f)

    @property
    def euler_angles(self) -> tuple:
        euler_angles = self._setup["Initial"][0]["EulerAngles"]
        phi0 = euler_angles[0]
        theta0 = euler_angles[1]
        psi0 = euler_angles[2]

        return phi0, theta0, psi0

    @property
    def rotation(self) -> tuple:
        omega = self._setup["Initial"][0]["rotVelocity"]
        p0 = omega[0]
        q0 = omega[1]
        r0 = omega[2]

        return p0, q0, r0

    @property
    def iterations_info(self) -> tuple:
        t0 = self._setup["iterations"][0]["start"]
        tend = self._setup["iterations"][0]["stop"]
        tstep = self._setup["iterations"][0]["step"]

        return t0, tend, tstep

    @property
    def torquer_params(self) -> tuple:
        n = self._setup["magnetorquer"][0]["n"]
        area = self._setup["magnetorquer"][0]["A"]

        return n, area

    @property
    def sat_params(self) -> tuple:
        inertia = self._setup["Satelite"][0]["I"]
        m = self._setup["Satelite"][0]["mass"]

        return inertia, m

    @property
    def planet_data(self) -> tuple:
        gravity_const = self._setup["PlanetConst"][0]["G"]
        earth_mass = self._setup["PlanetConst"][0]["M"]
        earth_radius = self._setup["PlanetConst"][0]["R"]

        return gravity_const, earth_mass, earth_radius
