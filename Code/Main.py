import pandapower as pp
import pandas as pd
from line_param_calc import calc_line

def initialize_net(path_bus, path_line, path_load):
    """
    initialize the grid from the .csv files

    :param path_bus: path to the bus .csv file
    :param path_line: path to the line .csv file
    :param path_load: path to the load .csv file
    :return: the net class
    """

    # create empty network
    net = pp.create_empty_network()

    # buses
    df_bus = pd.read_csv('Datafiles/bus1.csv')
    net.bus = df_bus

    # lines
    df_line = pd.read_csv('Datafiles/line1.csv')
    for _, line in df_line.iterrows():
        from_bus = pp.get_element_index(net, "bus", line.from_bus)
        to_bus = pp.get_element_index(net, "bus", line.to_bus)

        rr, xx, cc, imax = calc_line(line.a,
                                     line.b,
                                     line.c,
                                     line.d,
                                     line.e,
                                     line.max_i,
                                     int(line.parallel))

        pp.create_line_from_parameters(net,
                                       from_bus,
                                       to_bus,
                                       length_km=line.length,
                                       r_ohm_per_km=rr,
                                       x_ohm_per_km=xx,
                                       c_nf_per_km=cc,
                                       max_i_ka=imax,
                                       name=line.name_l,
                                       parallel=line.parallel)

    # loads
    df_load = pd.read_csv('Datafiles/load1.csv')

    # gens

    # trafos



    return net


if __name__ == "__main__":
    path_bus = 'Datafiles/bus1.csv'
    path_line = 'Datafiles/line1.csv'
    net = initialize_net(path_bus, path_line)
    print(net.bus)
    print(net.line)

# also add trafos and loads
