"""
Main module for ADCS simulation.

"""

import math
from dataclasses import dataclass

import igrf
import numpy as np
import pandas as pd

from adcs_simulation.configuration.config import Config
from adcs_simulation.propagator import eci_2_sb_transformation, Propagator
from adcs_simulation.visualization import visualize


@dataclass
class Result:
    """
    Class responsible for parsing results
    """
    pqr_table: pd.DataFrame
    state_vector: pd.DataFrame
    sun_table: pd.Series
    mag_field: pd.DataFrame
    angle: pd.DataFrame


def simulate(cfg: Config, propagator: Propagator) -> Result:
    """
    Main simulation code.

    The ADCS simulation is responsible to obtain orientation and
    position data of the satellite. Initial parameters such as
    time step control, initial Euler Angles and rotational
    velocity are loaded from setup.json file. For position
    the SGP4 propagator is utilized. To solve differential
    equations describing the satellites behavior the
    Runge-Kutta method (4th order) was used. It is an iterative
    algorithm to obtain approximate solutions.


    Parameters
    ----------
    cfg: Config - initial parameters from setup.json
    propagator: Propagator - data computed using the SGP4
        propagator such as: position and velocity in GCRS
        coordinate system, LLA (latitude, longitude, attitude),
        Sun vector in ECI frame and SB frame.

    Returns
    -------
    Results - appends the time step data to the following:
        pqr_table - data frame containing summary for different types of
            rotation velocity (magnetic, sun etc.)
        state_vector - data frame with summary information about
            the satellite, contains position, linear and rotation
            velocity, LLA
        sun_table - data frame with sun vector in ECI and SB frames
        mag_field - data frame with different types of magnetic field values
        angle - data frame with angle between two subsequent Sun vectors
    """

    tzero, tend, tstep = cfg.iterations_info
    state_vector, quaternions, mag_field, pqr_table, sun_table = initial_step(
        tzero, cfg, propagator
    )
    angle = {"phi": 0, "theta": 0, "psi": 0}
    angle = pd.DataFrame(data=angle, index=[0])
    #TODO: angle was a variable for debug, check what it is and if its necessary

    for delta_t in np.arange(tzero + 1, tend, tstep):
        print("Timestep {delta_t}")

        part_of_day = delta_t / (60 * 60 * 24)
        pos_gcrs, v_gcrs, lla, sun_sb, sun_eci = propagator.propagate(
            part_of_day, quaternions[-1]
        )

        pqr_sun, sun_table, angle = pqr_from_sun(sun_sb, sun_eci, sun_table)

        pqr_rk4, quaternion_rk4, mag_field, pqr_filt = rk4(
            delta_t,
            lla,
            state_vector.values[delta_t - 1],
            quaternions[delta_t - 1],
            mag_field,
            pqr_table,
            cfg,
        )

        pqr_table = update_pqr(pqr_rk4, pqr_filt, pqr_sun, pqr_table)
        quaternions, state_vector = update_data(
            state_vector, pos_gcrs, v_gcrs, lla, pqr_rk4, quaternions, quaternion_rk4
        )

    return Result(
        pqr_table=pqr_table,
        state_vector=state_vector,
        sun_table=sun_table,
        mag_field=mag_field,
        angle=angle,
    )


def initial_step(tzero: int, config: Config, propagator: Propagator):
    """
    Initialize the simulation for time zero.

    The first step starts from initial orientation parameters
    given in setup json. Propagation is executed for time t zero.

    In this part of the code where tables to store data during the
    simulation time steps are initialized. They are in form
    of pandas data frames.

    Parameters
    ----------
    tzero : int - simulation start time
    config : object - initial parameters from setup.json
    propagator: Propagator - data computed using the SGP4
        propagator such as: position and velocity in GCRS
        coordinate system, LLA (latitude, longitude, attitude),
        Sun vector in ECI frame and SB frame.

    Returns
    -------
    pqr_table - data frame containing rotation velocities of the satellite.
        Two based on magnetic field, measured (with bias), filtered
        (utilizing a simple filter algorithm) and one computed from
        the Sun vector change.
    state_vector - data frame with summary information about
        the satellite, contains position, linear and rotation
        velocity, LLA
    sun_table - data frame with sun vector in ECI and SB frames
    mag_field - data frame with magnetic field values, similarly to the
        pqr values one measured and one filtered. Both ale in ECI frame.

    """

    p0, q0, r0 = config.rotation
    phi0, theta0, psi0 = config.euler_angles

    quat0 = euler_to_quaternion(phi0, theta0, psi0)
    quaternions = [quat0]

    pos_gcrs, v_gcrs, lla, sun_sb, sun_eci = propagator.propagate(
        tzero / (60 * 60 * 24), quat0
    )
    col = {
        "x km": pos_gcrs[0],
        "y km": pos_gcrs[1],
        "z km": pos_gcrs[2],
        "dx/dt km/s": v_gcrs[0],
        "dy/dt km/s": v_gcrs[1],
        "dz/dt km/s": v_gcrs[2],
        "p rad/s": p0,
        "q rad/s": q0,
        "r rad/s": r0,
        "longtitude": lla[0].degrees,
        "latitude": lla[1].degrees,
        "altitude": lla[2],
    }
    state_vector = pd.DataFrame(data=col, index=[0])

    print("Evaluating initial time step for timestep {tzero}, state vector: \n")
    print(state_vector)
    print("\n")

    empty_b = {
        "B_x": 0.0,
        "B_y": 0.0,
        "B_z": 0.0,
        "B_x_filt": 0.0,
        "B_y_filt": 0.0,
        "B_z_filt": 0.0,
    }
    mag_field = pd.DataFrame(data=empty_b, index=[0])

    init_pqr = {
        "p": p0,
        "q": q0,
        "r": r0,
        "p_filt": p0,
        "q_filt": q0,
        "r_filt": r0,
        "p_sun": p0,
        "q_sun": q0,
        "r_sun": r0,
    }
    pqr_table = pd.DataFrame(data=init_pqr, index=[0])

    init_sun = {
        "X": sun_sb[0],
        "Y": sun_sb[1],
        "Z": sun_sb[2],
        "X_ECI": sun_eci[0],
        "Y_ECI": sun_eci[1],
        "Z_ECI": sun_eci[2],
    }
    sun_table = pd.DataFrame(data=init_sun, index=[0])

    return state_vector, quaternions, mag_field, pqr_table, sun_table


def pqr_from_sun(sun_sb: list, sun_eci: list, sun_table: pd.DataFrame):
    """
    Function to calculate the satellite rotational velocity from the Sun
    vector measurement.

    At first the function updates the sun_table by the Sun vector results
    from the current step. Then the te two last rows are taken to obtain
    the angle between them. Rotation velocity can be calculated as:
    Δangle/Δtime, as the time step is 1s this can be expressed directly
    as the angle between two iterations of the Sun vector


    Parameters
    ----------
    sun_sb : list - the x, y and z components of the Sun vector in SB frame.
        The vector is directly expressed as the position of the Sun in SB frame.
        It was transformed from ECI frame.
    sun_eci : list - the x, y and z components of the Sun vector in ECI frame.
        Also expressed as the position of the Sun in ECI frame.
    sun_table : pd.DataFrame - a frame containing the Sun vectors in ECI and
        SB frames from the previous steps

    Returns
    -------
    pqr : list - 3-element list with rotation velocity about all axes obtained
        from Sun vector
    sun_table : pd.DataFrame - a frame containing the Sun vectors in ECI and
        SB frames from the previous steps, now updated by the current step
    angle - the total angle between the two subsequent Sun vectors
    """

    sun_step = {
        "X": sun_sb[0],
        "Y": sun_sb[1],
        "Z": sun_sb[2],
        "X_ECI": sun_eci[0],
        "Y_ECI": sun_eci[1],
        "Z_ECI": sun_eci[2],
    }
    df2 = pd.DataFrame(data=sun_step, index=[0])
    sun_table = pd.concat([sun_table, df2])

    s_vector0 = sun_table.values[-2]
    s_vector0 = s_vector0[:3] / np.linalg.norm(s_vector0[:3])
    s_vector1 = sun_table.values[-1]
    s_vector1 = s_vector1[:3] / np.linalg.norm(s_vector1[:3])

    if sum(s_vector0) == 0 or sum(s_vector1) == 0:
        pqr = [0, 0, 0]

    else:
        dot_product = np.dot(s_vector0, s_vector1)
        cross_product = np.cross(s_vector0, s_vector1)
        cross_product = (
                -1 * cross_product / np.linalg.norm(cross_product)
        )  # nie wiem czemu razy -1 ale dziala
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

        deltaphi = angle * cross_product[0]
        deltatheta = angle * cross_product[1]
        deltapsi = angle * cross_product[2]

        # assumption: if dt = 1s -> p ~ detlaphi, q ~ deltatheta, r ~ deltapsi
        pqr = [deltaphi, deltatheta, deltapsi]

    return pqr, sun_table, angle


def update_pqr(pqr_rk4: np.array, pqr_filt: np.array,
                 pqr_sun: list, pqr_table: pd.DataFrame) -> pd.DataFrame:
    """
    Update the sun_table data frame by calculated in the current step values
    of different rotational velocities.

    Parameters
    ----------
    pqr_rk4 : np.array - rotational velocity calculated using the
        Runge-Kutte algorithm based on magnetic field measurement
    pqr_filt : np.array - rotational velocity computed from the previous
        RK4 version first by adding bias the applying a filter.
    pqr : list - 3-element list with rotation velocity about all axes obtained
        from Sun vector
    pqr_table : DataFrame - data frame containing rotation velocities of the satellite.
        Two based on magnetic field, measured (with bias), filtered
        (utilizing a simple filter algorithm) and one computed from
        the Sun vector change.

    Returns
    -------
    pqr_table: DataFrame - data frame containing rotation velocities of the satellite.
        Two based on magnetic field, measured (with bias), filtered
        (utilizing a simple filter algorithm) and one computed from
        the Sun vector change. Now updated by the current step.
    """

    step_pqr = {
        "p": pqr_rk4[0],
        "q": pqr_rk4[1],
        "r": pqr_rk4[2],
        "p_filt": pqr_filt[0],
        "q_filt": pqr_filt[1],
        "r_filt": pqr_filt[2],
        "p_sun": pqr_sun[0],
        "q_sun": pqr_sun[1],
        "r_sun": pqr_sun[2],
    }
    df2 = pd.DataFrame(data=step_pqr, index=[0])
    pqr_table = pd.concat([pqr_table, df2])

    return pqr_table


def euler_to_quaternion(phi: float, theta: float, psi: float) -> list[float]:
    """
    Mathematical function to transform the Euler angles rotation
    representation into a quaternion representation.

    Parameters
    ----------
    phi : float - rotation around the X Axis in radians
    theta : float - rotation around the Y Axis in radians
    psi : float - rotation around the Z Axis in radians

    Returns
    -------
    qauternion : list - quaternion representation of the given Euler angles. The initial
        orientation is given as the axis of ECI (Earth Centered Inertial) frame.
        A quaternion contains four elements (here in form of a list). Three
        of the elements are vectors and only one is a scalar:
        q = iq1 + jq2 + kq3 + q4,
        where i, j and are describing the vector which is the rotation axis
        The q1 to q3 can be written as sin(alfa/2) and q4 cos(alfa/2),
        where alfa is the angle of rotation around the given axis.
    """

    #TODO: check the input angles (radians or degrees)


    qat0 = math.cos(phi / 2) * math.cos(theta / 2) * math.cos(psi / 2) + math.sin(
        phi / 2) * math.sin(theta / 2) * math.sin(psi / 2)
    qat1 = math.sin(phi / 2) * math.cos(theta / 2) * math.cos(psi / 2) - math.cos(
        phi / 2) * math.sin(theta / 2) * math.sin(psi / 2)
    qat2 = math.cos(phi / 2) * math.sin(theta / 2) * math.cos(psi / 2) + math.sin(
        phi / 2) * math.cos(theta / 2) * math.sin(psi / 2)
    qat3 = math.cos(phi / 2) * math.cos(theta / 2) * math.sin(psi / 2) - math.sin(
        phi / 2) * math.sin(theta / 2) * math.cos(psi / 2)

    qauternion = [qat0, qat1, qat2, qat3]

    return qauternion


def ned_2_eci_transformation(ned: list, xyz: np.array) -> np.array:
    """
    Function to transform vector given in NED (North-East-Down) frame to
    ECI (Earth Centered Inertial) frame.

    Parameters
    ----------
    ned : list - 3D vector in NED frame. In the NED frame X Axis points to
        true North, Z Axis points towards the Earth center, Y Axis completes
        the right-handed system (points East)
    xyz : np.array - X, Y and Z coordinates of the object given as a
        vector in ECI frame

    Returns
    -------
    eci : np.array - input vector transformed into ECI frame. Dot product of
    the computed rotation matrix and input vector
    """

    #TODO rewrite using scipy rotation library
    rho = np.linalg.norm(xyz)
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]

    phiE = 0
    thetaE = np.arccos(z / rho) + np.pi
    psiE = np.arctan2(y, x)

    cost = np.cos(thetaE)
    sint = np.sin(thetaE)
    sinp = np.sin(phiE)
    cosp = np.cos(phiE)
    sins = np.sin(psiE)
    coss = np.cos(psiE)

    rot_mat = [
        [cost * coss, sinp * sint * coss - cosp * sins, cosp * sint * coss + sinp * sins],
        [cost * sins, sinp * sint * sins + cosp * coss, cosp * sint * sins - sinp * coss],
        [-sint, sinp * cost, cosp * cost],
    ]
    eci = np.dot(rot_mat, ned)

    return eci


def quaternion_diff(pqr: np.array, quaternion: list[float]):
    """
    Mathematical function to obtain the the differential of a quaternion
    dq/dt for a given rotation with the given rotation velocity.

    Parameters
    ----------
    pqr : np.array - array whit the current rotation velocity of the object
        in radians in SB (Satellite Body) frame
    quaternion : list[float] - quaternion describing the rotation from the
        ECI axis to the current orientation

    Returns
    -------
    quaternion_dot : np.array - differential quaternion
    """
    p = pqr[0]
    q = pqr[1]
    r = pqr[2]

    prq_matrix = np.array(
        [[0, -p, -q, -r], [p, 0, r, -q], [q, -r, 0, p], [r, q, -p, 0]]
    )
    quat = np.array(quaternion)
    quaternion_dot = np.dot(0.5 * prq_matrix, quat)

    return quaternion_dot


def polute_signal(b_sb: np.array, pqr: np.array) -> np.array:
    """
    Short function to pollute the magnetic field and rotation velocity
    calculation to imitate real measurements.

    Parameters
    ----------
    b_sb : np.array - calculated magnetic field vector given in the SB
        (Satellite Body) frame
    pqr : np.array - calculated rotation velocity vector given in the SB
        (Satellite Body) frame

    Returns
    -------
    polluted_b_SB : np.array - the input magnetic field updated by bias and noise
    polluted_pqr : np.array - the input rotation velocity updated by bias
        and noise
    """

    #TODO: random values from -1 to 1? why 0 to 1 * 2 used?
    # Bias and Noise
    magscale_bias = 4e-7  # T
    mag_field_bias = magscale_bias * (2 * np.random.rand())  # (0 to 1)

    magscale_noise = 1e-5  # T
    mag_field_noise = magscale_noise * (2 * np.random.rand())  # (0 to 1)

    angscale_bias = 0.01  # rad/s
    ang_field_bias = angscale_bias * (2 * np.random.rand())  # (0 to 1)

    angscale_noise = 0.001  # rad/s
    AngFieldNoise = angscale_noise * (2 * np.random.rand())  # (0 to 1)

    polluted_pqr = np.zeros(3)
    polluted_b_SB = np.zeros(3)

    for x in range(3):
        polluted_b_SB[x] = b_sb[x] + mag_field_bias + mag_field_noise
        polluted_pqr[x] = pqr[x] + ang_field_bias + AngFieldNoise

    return np.array(polluted_b_SB), np.array(polluted_pqr)


def filter_signal(b_poluted: np.array, pqr_poluted: np.array,
            mag_field: pd.DataFrame, pqr_table: pd.DataFrame) -> np.array:
    """
    Simple filter for the polluted data.

    Takes the previous data for magnetic field and rotation velocity and the
    current values which were polluted to simulate a real measurement.
    As we assume rather small values and small changes in time the new filtered
    value is taken as a sum of parts of the previous value and measured:
    new value = current * s + previous * (1-s),
    where we rather thrust the previous values default using s = 0.3 to
    avoid step changes from measurement errors.
    Estimated bias should be also subtracted from the measured values, but
    at this time we assume them as 0.

    Parameters
    ----------
    b_poluted : np.array - magnetic field in SB (Satellite Body) frame with
        added pollution to simulate real measurements
    pqr_poluted : np.array - rotation velocity in SB (Satellite Body) frame
        with added pollution to simulate real measurements
    mag_field : pd.DataFrame - table with saved magnetic field data to get
        the values from the previous step
    pqr_table : pd.DataFrame - table with saved rotation velocity data to get
        the values from the previous step

    Returns
    -------
    b_field_nav : np.array - array with the filtered (navigation)
        magnetic field
    pqr_nav : np.array - array with the filtered (navigation)
        rotation velocity
    """
    if all(mag_field[-1]) == 0:
        b_field_nav = b_poluted
        pqr_nav = pqr_poluted
    else:
        B_prev = np.array(mag_field[-1][3:])
        pqr_prev = np.array(pqr_table[-1][3:6])

        s = 0.3

        BiasEstimate = [0, 0, 0]
        b_field_nav = B_prev * (1 - s) + s * (b_poluted - BiasEstimate)
        pqrBiasEstimate = [0, 0, 0]
        pqr_nav = pqr_prev * (1 - s) + s * (pqr_poluted - pqrBiasEstimate)

    return b_field_nav, pqr_nav


def magnetorquer(b_sb_filt, pqr_filt, config: Config):
    n, area = config.torquer_params
    k = 67200
    current = k * np.cross(pqr_filt, b_sb_filt) / (n * area)

    # Add in saturation
    if np.abs(np.max(current)) > 0.04:
        current = current / np.linalg.norm(current) * 0.04
    mu_b = current * n * area

    return current, mu_b


def rot_dynamics(muB, B_SB, pqr, config: Config):
    I, _ = config.sat_params

    torquer_model = np.dot(muB, B_SB)
    H = np.dot(I, pqr)
    invI = np.linalg.inv(np.array(I))
    pqr = np.array(pqr)

    pqr_dot = np.dot(np.array(invI), (torquer_model - np.cross(pqr, H)))
    return pqr_dot


def trans_dynamics(xyz, config: Config):
    I, m = config.sat_params
    G, M, R = config.planet_data

    rho = np.linalg.norm(xyz)
    rhat = xyz / rho
    force_grav = -(G * M * m / rho ** 2) * rhat

    accel = force_grav / m

    return accel


def k_function(
        t, lla, quaternion, pqr, xyz, mag_field, pqr_table, config: Config, rewrite
):
    quaternion_dot = quaternion_diff(pqr, quaternion)
    if t % 10 == 0:
        mag = igrf.igrf(
            "2020-01-01", glat=lla[0].degrees, glon=lla[1].degrees, alt_km=lla[2]
        )

        B_NED = [mag["north"][0], mag["east"][0], mag["down"][0]]
        B_ECI = ned_2_eci_transformation(B_NED, xyz)
        b_SB = eci_2_sb_transformation(B_ECI, quaternion)
        b_SB = b_SB * 1e-9
    else:
        b_SB = mag_field.values[-1][:3]

    if t % 10 == 0 and rewrite == True:
        B_SB_Measured, pqr_measured = polute_signal(b_SB, pqr)
        B_SB_filt, pqr_filt = filter_signal(
            B_SB_Measured, pqr_measured, mag_field.values, pqr_table.values
        )
        B_step = {
            "B_x": b_SB[0],
            "B_y": b_SB[1],
            "B_z": b_SB[2],
            "B_x_filt": B_SB_filt[0],
            "B_y_filt": B_SB_filt[1],
            "B_z_filt": B_SB_filt[2],
        }
        df2 = pd.DataFrame(data=B_step, index=[0])
        mag_field = pd.concat([mag_field, df2])

        pqr_step = {
            "p": pqr[0],
            "q": pqr[1],
            "r": pqr[2],
            "p_filt": pqr_filt[0],
            "q_filt": pqr_filt[1],
            "r_filt": pqr_filt[2],
        }
        df2 = pd.DataFrame(data=pqr_step, index=[0])
        pqr_table = pd.concat([pqr_table, df2])

    elif t % 10 != 0 and rewrite == True:
        B_SB_filt = mag_field.values[-1][3:6]
        pqr_filt = pqr_table.values[-1][3:6]
        B_step = {
            "B_x": b_SB[0],
            "B_y": b_SB[1],
            "B_z": b_SB[2],
            "B_x_filt": B_SB_filt[0],
            "B_y_filt": B_SB_filt[1],
            "B_z_filt": B_SB_filt[2],
        }
        df2 = pd.DataFrame(data=B_step, index=[0])
        mag_field = pd.concat([mag_field, df2])
        pqr_step = {
            "p": pqr[0],
            "q": pqr[1],
            "r": pqr[2],
            "p_filt": pqr_filt[0],
            "q_filt": pqr_filt[1],
            "r_filt": pqr_filt[2],
        }
        df2 = pd.DataFrame(data=pqr_step, index=[0])
        pqr_table = pd.concat([pqr_table, df2])
    else:
        B_SB_filt = mag_field.values[-1][3:6]
        pqr_filt = pqr_table.values[-1][3:6]
        B_step = {
            "B_x": b_SB[0],
            "B_y": b_SB[1],
            "B_z": b_SB[2],
            "B_x_filt": B_SB_filt[0],
            "B_y_filt": B_SB_filt[1],
            "B_z_filt": B_SB_filt[2],
        }
        pqr_step = {
            "p": pqr[0],
            "q": pqr[1],
            "r": pqr[2],
            "p_filt": pqr_filt[0],
            "q_filt": pqr_filt[1],
            "r_filt": pqr_filt[2],
        }

    current, muB = magnetorquer(B_SB_filt, pqr_filt, config)
    pqr_dot = rot_dynamics(muB, b_SB, pqr, config)

    dstatdt = np.array(
        [
            quaternion_dot[0],
            quaternion_dot[1],
            quaternion_dot[2],
            quaternion_dot[3],
            pqr_dot[0],
            pqr_dot[1],
            pqr_dot[2],
        ]
    )

    return dstatdt, mag_field, pqr_filt


def rk4(t, lla, state_vector, quaternion, mag_field, pqr_table, config: Config):
    t0, tend, tstep = config.iterations_info

    pqr = np.array([state_vector[6], state_vector[7], state_vector[8]])
    xyz = np.array([state_vector[0], state_vector[1], state_vector[2]])
    v_xyz = np.array([state_vector[3], state_vector[4], state_vector[5]])
    k1, mag_field, pqr_filt = np.array(
        k_function(
            t, lla, quaternion, pqr, xyz, mag_field, pqr_table, config, rewrite=True
        ),
        dtype=object,
    )
    k2, mag_field, temp = np.array(
        k_function(
            t + tstep / 2,
            lla,
            quaternion + k1[0:4] * tstep / 2,
            pqr + k1[4:7] * tstep / 2,
            xyz + v_xyz * tstep / 2,
            mag_field,
            pqr_table,
            config,
            rewrite=False,
        ),
        dtype=object,
    )
    k3, mag_field, temp = np.array(
        k_function(
            t + tstep / 2,
            lla,
            quaternion + k2[0:4] * tstep / 2,
            pqr + k2[4:7] * tstep / 2,
            xyz + v_xyz * tstep / 2,
            mag_field,
            pqr_table,
            config,
            rewrite=False,
        ),
        dtype=object,
    )
    k4, mag_field, temp = np.array(
        k_function(
            t + tstep,
            lla,
            quaternion + k3[0:4] * tstep,
            pqr + k3[4:7] * tstep,
            xyz + v_xyz * tstep,
            mag_field,
            pqr_table,
            config,
            rewrite=False,
        ),
        dtype=object,
    )

    k = np.array((1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4))
    pqr1 = pqr + k[4:7] * tstep
    quaternion1 = quaternion + k[0:4] * tstep

    return pqr1, quaternion1, mag_field, pqr_filt


def update_data(state_vector, pos_GCRS, v_GCRS, lla, pqr_rk4, quaternions, quaternion):
    col = {
        "x km": pos_GCRS[0],
        "y km": pos_GCRS[1],
        "z km": pos_GCRS[2],
        "dx/dt km/s": v_GCRS[0],
        "dy/dt km/s": v_GCRS[1],
        "dz/dt km/s": v_GCRS[2],
        "p rad/s": pqr_rk4[0],
        "q rad/s": pqr_rk4[1],
        "r rad/s": pqr_rk4[2],
        "longtitude": lla[0].degrees,
        "latitude": lla[1].degrees,
        "altitude": lla[2],
    }
    temp_state_vector = pd.DataFrame(data=col, index=[0])
    state_vector = pd.concat([state_vector, temp_state_vector])

    quaternions.append(quaternion)
    return quaternions, state_vector


def main():
    cfg = Config("ADCS_setup.json")
    propagator = Propagator("tle.txt")
    result = simulate(cfg, propagator)
    G, M, R = cfg.planet_data
    visualize(result, R)


if __name__ == "__main__":
    main()
