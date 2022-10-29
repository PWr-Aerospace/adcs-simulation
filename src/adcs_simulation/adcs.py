"""
Main module for ADCS simulation.

"""

import math
from dataclasses import dataclass

import igrf
import numpy as np
import pandas as pd

from adcs_simulation.configuration.config import Config
from adcs_simulation.propagator import ECI_2_SB, Propagator
from adcs_simulation.visualization import visualize


@dataclass
class Result:
    pqr_table: pd.DataFrame
    state_vector: pd.DataFrame
    sun_table: pd.Series
    mag_field: pd.DataFrame
    angle: pd.DataFrame


def simulate(cfg, propagator) -> Result:
    t0, tend, tstep = cfg.iterations_info
    state_vector, quaternions, mag_field, pqr_table, sun_table = initial_step(
        t0, cfg, propagator
    )
    angle = {"cos_p": 0, "cos_q": 0, "cos_r": 0}
    angle = pd.DataFrame(data=angle, index=[0])

    for dt in np.arange(t0 + 1, tend, tstep):
        print("Timestep {}".format(dt))

        part_of_day = dt / (60 * 60 * 24)
        pos_GCRS, v_GCRS, lla, sun_SB, sun_ECI = propagator.propagate(
            part_of_day, quaternions[-1]
        )

        pqr_sun, sun_table, angle = pqr_from_sun(sun_SB, sun_ECI, sun_table, angle)

        pqr_rk4, quaternion_rk4, mag_field, pqr_filt = rk4(
            dt,
            lla,
            state_vector.values[dt - 1],
            quaternions[dt - 1],
            mag_field,
            pqr_table,
            cfg,
        )

        pqr_table = update_pqr(pqr_rk4, pqr_filt, pqr_sun, pqr_table)
        quaternions, state_vector = update_data(
            state_vector, pos_GCRS, v_GCRS, lla, pqr_rk4, quaternions, quaternion_rk4
        )

    return Result(
        pqr_table=pqr_table,
        state_vector=state_vector,
        sun_table=sun_table,
        mag_field=mag_field,
        angle=angle,
    )


def initial_step(t0, config: Config, propagator: Propagator):
    p0, q0, r0 = config.rotation
    phi0, theta0, psi0 = config.euler_angles

    quat0 = Euler_to_quaternion(phi0, theta0, psi0)
    quaternions = [quat0]

    pos_GCRS, v_GCRS, lla, sun_SB, sun_ECI = propagator.propagate(
        t0 / (60 * 60 * 24), quat0
    )
    col = {
        "x km": pos_GCRS[0],
        "y km": pos_GCRS[1],
        "z km": pos_GCRS[2],
        "dx/dt km/s": v_GCRS[0],
        "dy/dt km/s": v_GCRS[1],
        "dz/dt km/s": v_GCRS[2],
        "p rad/s": p0,
        "q rad/s": q0,
        "r rad/s": r0,
        "longtitude": lla[0].degrees,
        "latitude": lla[1].degrees,
        "altitude": lla[2],
    }
    state_vector = pd.DataFrame(data=col, index=[0])

    print("Evaluating initial time step for timestep {}, state vector: \n".format(t0))
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
        "X": sun_SB[0],
        "Y": sun_SB[1],
        "Z": sun_SB[2],
        "X_ECI": sun_ECI[0],
        "Y_ECI": sun_ECI[1],
        "Z_ECI": sun_ECI[2],
    }
    sun_table = pd.DataFrame(data=init_sun, index=[0])

    return state_vector, quaternions, mag_field, pqr_table, sun_table


def pqr_from_sun(sun_SB, sun_ECI, sun_table, agle):

    sun_step = {
        "X": sun_SB[0],
        "Y": sun_SB[1],
        "Z": sun_SB[2],
        "X_ECI": sun_ECI[0],
        "Y_ECI": sun_ECI[1],
        "Z_ECI": sun_ECI[2],
    }
    df2 = pd.DataFrame(data=sun_step, index=[0])
    sun_table = pd.concat([sun_table, df2])

    u0 = sun_table.values[-2]
    u0 = u0[:3] / np.linalg.norm(u0[:3])
    u1 = sun_table.values[-1]
    u1 = u1[:3] / np.linalg.norm(u1[:3])

    if sum(u0) == 0 or sum(u1) == 0:
        pqr = [0, 0, 0]

    else:
        dot_product = np.dot(u0, u1)
        cross_product = np.cross(u0, u1)
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


def update_pqr(pqr_rk4, pqr_filt, pqr_sun, pqr_table):

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


def Euler_to_quaternion(phi, theta, psi) -> float:

    q0 = math.cos(phi / 2) * math.cos(theta / 2) * math.cos(psi / 2) + math.sin(
        phi / 2
    ) * math.sin(theta / 2) * math.sin(psi / 2)
    q1 = math.sin(phi / 2) * math.cos(theta / 2) * math.cos(psi / 2) - math.cos(
        phi / 2
    ) * math.sin(theta / 2) * math.sin(psi / 2)
    q2 = math.cos(phi / 2) * math.sin(theta / 2) * math.cos(psi / 2) + math.sin(
        phi / 2
    ) * math.cos(theta / 2) * math.sin(psi / 2)
    q3 = math.cos(phi / 2) * math.cos(theta / 2) * math.sin(psi / 2) - math.sin(
        phi / 2
    ) * math.sin(theta / 2) * math.cos(psi / 2)

    q = [q0, q1, q2, q3]

    return q


def NED_2_ECI(NED, xyz):

    rho = np.linalg.norm(xyz)
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]

    phiE = 0
    thetaE = np.arccos(z / rho) + np.pi
    psiE = np.arctan2(y, x)

    ct = np.cos(thetaE)
    st = np.sin(thetaE)
    sp = np.sin(phiE)
    cp = np.cos(phiE)
    ss = np.sin(psiE)
    cs = np.cos(psiE)

    rot_mat = [
        [ct * cs, sp * st * cs - cp * ss, cp * st * cs + sp * ss],
        [ct * ss, sp * st * ss + cp * cs, cp * st * ss - sp * cs],
        [-st, sp * ct, cp * ct],
    ]
    ECI = np.dot(rot_mat, NED)

    return ECI


def quaternion_diff(pqr, quaternion):
    p = pqr[0]
    q = pqr[1]
    r = pqr[2]

    prq_matrix = np.array(
        [[0, -p, -q, -r], [p, 0, r, -q], [q, -r, 0, p], [r, q, -p, 0]]
    )
    quat = np.array(quaternion)
    quaternion_dot = np.dot(0.5 * prq_matrix, quat)

    return quaternion_dot


def polute_signal(b_SB, pqr):
    # Bias and Noise
    MagscaleBias = 4e-7  # T
    MagFieldBias = MagscaleBias * (2 * np.random.rand())  # (0 to 1) (0 to 2)

    MagscaleNoise = 1e-5  # T
    MagFieldNoise = MagscaleNoise * (2 * np.random.rand())  # (0 to 1) (0 to 2)

    AngscaleBias = 0.01  # rad/s
    AngFieldBias = AngscaleBias * (2 * np.random.rand())  # (0 to 1) (0 to 2)

    AngscaleNoise = 0.001  # rad/s
    AngFieldNoise = AngscaleNoise * (2 * np.random.rand())  # (0 to 1) (0 to 2)

    poluted_pqr = np.zeros(3)

    for x in range(3):
        b_SB[x] = b_SB[x] + MagFieldBias + MagFieldNoise
        poluted_pqr[x] = pqr[x] + AngFieldBias + AngFieldNoise

    return np.array(b_SB), np.array(poluted_pqr)


def filter_signal(B_poluted, pqr_poluted, mag_field, pqr_table):

    if all(mag_field[-1]) == 0:
        B_field_nav = B_poluted
        pqr_nav = pqr_poluted
    else:
        B_prev = np.array(mag_field[-1][3:])
        pqr_prev = np.array(pqr_table[-1][3:6])

        s = 0.3

        BiasEstimate = [0, 0, 0]
        B_field_nav = B_prev * (1 - s) + s * (B_poluted - BiasEstimate)
        pqrBiasEstimate = [0, 0, 0]
        pqr_nav = pqr_prev * (1 - s) + s * (pqr_poluted - pqrBiasEstimate)

    return B_field_nav, pqr_nav


def magnetorquer(B_SB_filt, pqr_filt, config: Config):
    n, A = config.torquer_params
    k = 67200
    current = k * np.cross(pqr_filt, B_SB_filt) / (n * A)

    # Add in saturation
    if np.abs(np.max(current)) > 0.04:
        current = current / np.linalg.norm(current) * 0.04
    muB = current * n * A

    return current, muB


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
    F_grav = -(G * M * m / rho**2) * rhat

    accel = F_grav / m

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
        B_ECI = NED_2_ECI(B_NED, xyz)
        b_SB = ECI_2_SB(B_ECI, quaternion)
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
