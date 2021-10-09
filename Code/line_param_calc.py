import numpy as np


def single_line(a, b):
    """
    calculate the R, X, C parameters, also return Imax

    :param a: horizontal distance between A and C
    :param b: vertical distance between A and B
    :return: R, X, C, Imax, in the units desired by pandapower
    """

    # cardinal: https://www.elandcables.com/media/38193/acsr-astm-b-aluminium-conductor-steel-reinforced.pdf
    # 54 Al + 7 St, Imax = 888.98 A

    w = 2 * np.pi * 50  # rad / s
    Imax = 888.98e-3  # kA
    Stot = 547.3 * 1e-6  # m2, the total section
    R_ac_75 = 0.07316 * 1e-3  # ohm / m
    kg = 0.809  # from the slides in a 54 + 7

    r = np.sqrt(Stot / np.pi)  # considering the total section

    dab = np.sqrt((a / 2) ** 2 + b ** 2)
    dbc = np.sqrt((a / 2) ** 2 + b ** 2)
    dca = a

    GMD = (dab * dbc * dca) ** (1 / 3)
    GMR = kg * r
    RMG = r

    L = 4 * np.pi * 1e-7 / (2 * np.pi) * np.log(GMD / GMR)  # H / m

    C = 2 * np.pi * 1e-9 / (36 * np.pi) / np.log(GMD / RMG)  # F / m

    # in the units pandapower wants
    R_km = R_ac_75 * 1e3  # ohm / km
    X_km = L * w * 1e3  # ohm / km
    C_km = C * 1e9 * 1e3  # nF / km

    return R_km, X_km, C_km, Imax


def double_line(a, b, c, d, e):
    """
    calculate the R, X, C parameters, also return Imax

    :param a: horizontal distance between A1 and C2
    :param b: horizontal distance between B1 and B2
    :param c: horizontal distance between C1 and A2
    :param d: vertical distance between A1 and B1
    :param e: vertical distance between B1 and C1
    :return: R, X, C, Imax, in the units desired by pandapower
    """

    # cardinal: https://www.elandcables.com/media/38193/acsr-astm-b-aluminium-conductor-steel-reinforced.pdf
    # 54 Al + 7 St, Imax = 888.98 A

    w = 2 * np.pi * 50  # rad / s
    Imax = 888.98e-3 * 2  # kA, for the full line, x2
    Stot = 547.3 * 1e-6  # m2, the total section
    R_ac_75 = 0.07316 * 1e-3  # ohm / m
    kg = 0.809  # from the slides in a 54 + 7

    r = np.sqrt(Stot / np.pi)  # considering the total section

    da1b1 = np.sqrt((b / 2 - a / 2) ** 2 + d ** 2)
    da1b2 = np.sqrt((a / 2 + b / 2) ** 2 + d ** 2)
    da2b1 = np.sqrt((c / 2 + b / 2) ** 2 + e ** 2)
    da2b2 = np.sqrt((b / 2 - c / 2) ** 2 + e ** 2)

    db1c1 = np.sqrt((b / 2 - c / 2) ** 2 + e ** 2)
    db1c2 = np.sqrt((b / 2 + a / 2) ** 2 + d ** 2)
    db2c1 = np.sqrt((b / 2 + c / 2) ** 2 + e ** 2)
    db2c2 = np.sqrt((b / 2 - a / 2) ** 2 + d ** 2)

    dc1a1 = np.sqrt((a / 2 - c / 2) ** 2 + (d + e) ** 2)
    dc1a2 = c
    dc2a1 = a
    dc2a2 = np.sqrt((a / 2 - c / 2) ** 2 + (d + e) ** 2)

    dab = (da1b1 * da1b2 * da2b1 * da2b2) ** (1 / 4)
    dbc = (db1c1 * db1c2 * db2c1 * db2c2) ** (1 / 4)
    dca = (dc1a1 * dc1a2 * dc2a1 * dc2a2) ** (1 / 4)

    rp = kg * r

    da1a2 = np.sqrt((a / 2 + c / 2) ** 2 + (d + e) ** 2)
    db1b2 = b
    dc1c2 = np.sqrt((c / 2 + a / 2) ** 2 + (d + e) ** 2)

    drap = np.sqrt(rp * da1a2)
    drbp = np.sqrt(rp * db1b2)
    drcp = np.sqrt(rp * dc1c2)

    dra = np.sqrt(r * da1a2)
    drb = np.sqrt(r * db1b2)
    drc = np.sqrt(r * dc1c2)

    GMD = (dab * dbc * dca) ** (1 / 3)
    GMR = (drap * drbp * drcp) ** (1 / 3)
    RMG = (dra * drb * drc) ** (1 / 3)

    L = 4 * np.pi * 1e-7 / (2 * np.pi) * np.log(GMD / GMR)  # H / m

    C = 2 * np.pi * 1e-9 / (36 * np.pi) / np.log(GMD / RMG)  # F / m

    # in the units pandapower wants
    R_km = R_ac_75 / 2 * 1e3  # ohm / km, like 2 resistances in parallel
    X_km = L * w * 1e3  # ohm / km
    C_km = C * 1e9 * 1e3  # nF / km

    return R_km, X_km, C_km, Imax


rr, xx, cc, ii = double_line(11, 2, 4, 5, 6)
print(rr, xx, cc, ii)

