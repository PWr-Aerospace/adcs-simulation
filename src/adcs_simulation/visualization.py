"""
Module for visualising data.

"""

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def visualize(result: Any, radius: float):
    """
    Function dedicated to plot the most important parameters

    Created plots:
        - 3D plot of the orbit in ECEF (Earth Centered Earth Fixed) frame.
        - Rotation velocity calculated from magnetic field.
        - Rotation velocity calculated from Sun vector.
        - Measured magnetic field.

    Parameters
    ----------
    results
    radius : float - Earth
    """

    sun_table = result.sun_table
    angle = result.angle
    state_vector = result.state_vector
    mag_field = result.mag_field
    pqr_table = result.pqr_table

    sun_table.to_csv("sun_table.csv", index=False)
    angle.to_csv("angles.csv", index=False)
    print(state_vector)
    print(mag_field)
    print(pqr_table)

    os.mkdir("../../plots")

    fig = plt.figure()
    axis = fig.gca(projection="3d")
    axis.set_aspect("auto")
    # draw sphere
    u, v = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]
    x_coordinate = np.cos(u) * np.sin(v) * radius / 1000
    y_coordinate = np.sin(u) * np.sin(v) * radius / 1000
    z_coordinate = np.cos(v) * radius / 1000

    axis.title("Orbit ECEF")
    # ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r)
    axis.plot_wireframe(x_coordinate, y_coordinate, z_coordinate, color="g")
    axis.scatter(
        state_vector["x km"],
        state_vector["y km"],
        state_vector["z km"],
        color="r",
        s=20,
    )
    plt.savefig()
    plt.show()
    plt.close("../../plots/orbit_ECEF")

    plt.title("Position ECEF")
    plt.grid()
    plt.plot(range(len(state_vector["x km"])), state_vector["x km"])
    plt.plot(range(len(state_vector["x km"])), state_vector["y km"])
    plt.plot(range(len(state_vector["x km"])), state_vector["z km"])
    plt.xlabel("time [s]")
    plt.ylabel("position [km]")
    plt.legend(["x", "y", "z"], loc="best")
    plt.savefig("../../plots/position_ECEF.png", dpi=720)

    plt.show()
    plt.close()

    plt.title("Satellite body rotation (magnetic field)")
    plt.grid()
    plt.plot(range(len(state_vector["x km"])), pqr_table["p"], label="p")
    plt.plot(range(len(state_vector["x km"])), pqr_table["q"], label="q")
    plt.plot(range(len(state_vector["x km"])), pqr_table["r"], label="r")
    plt.plot(
        range(len(state_vector["x km"])),
        np.sqrt(
            state_vector["p rad/s"] ** 2
            + state_vector["q rad/s"] ** 2
            + state_vector["r rad/s"] ** 2
        ),
        label="omega",
    )
    plt.xlabel("time [s]")
    plt.ylabel("rotation [rad/s]")
    plt.legend(loc="best")
    plt.savefig("../../plots/mag_rotation.png", dpi=720)

    plt.show()
    plt.close()

    plt.title("Satliette body rotation (sun vector)")
    plt.grid()
    plt.plot(range(len(state_vector["x km"])), pqr_table["p_sun"], label="p")
    plt.plot(range(len(state_vector["x km"])), pqr_table["q_sun"], label="q")
    plt.plot(range(len(state_vector["x km"])), pqr_table["r_sun"], label="r")
    plt.plot(
        range(len(state_vector["x km"])),
        np.sqrt(
            pqr_table["p_sun"] ** 2 + pqr_table["q_sun"] ** 2 + pqr_table["r_sun"] ** 2
        ),
        label="omega",
    )
    plt.xlabel("time [s]")
    plt.ylabel("rotation [rad/s]")
    plt.legend(loc="best")
    plt.savefig("../../plots/sun_rotation.png", dpi=720)

    plt.show()
    plt.close()

    plt.title("Magnetic field SBRF")
    plt.grid()
    plt.plot(range(len(state_vector["x km"])), mag_field["B_x"], label="X")
    plt.plot(range(len(state_vector["x km"])), mag_field["B_y"], label="Y")
    plt.plot(range(len(state_vector["x km"])), mag_field["B_z"], label="Z")
    plt.plot(
        range(len(state_vector["x km"])),
        np.sqrt(mag_field["B_z"] ** 2 + mag_field["B_y"] ** 2 + mag_field["B_x"] ** 2),
        label="Summary",
    )
    plt.xlabel("time [s]")
    plt.ylabel("Magnetic field [T]")
    plt.legend(loc="best")
    plt.savefig("../../plots/mag_field.png", dpi=720)

    plt.show()
    plt.close()
