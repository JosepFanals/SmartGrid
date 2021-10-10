import pandapower as pp
import pandas as pd
import numpy as np
import pandapower.control as control
import pandapower.networks as nw
import pandapower.timeseries as timeseries
from pandapower.timeseries.data_sources.frame_data import DFData
from line_param_calc import calc_line

def initialize_net(path_bus, path_line, path_demand, path_busload):
    """
    initialize the grid from the .csv files

    :param path_bus: path to the bus .csv file
    :param path_line: path to the line .csv file
    :param path_demand: path to the normalized demand .csv file
    :param busload: path to the bus-load look up table .csv file
    :return: the net class
    """

    def create_bus(path_bus):
        """
        adapts the data from the bus file (if needed)

        :param path_bus:
        :return: the net with the buses added
        """

        df_bus = pd.read_csv(path_bus)
        net.bus = df_bus
        return net

    def create_line(path_line):
        """
        adapts the data from the line file

        :param path_line:
        :return: the net with the lines added
        """

        df_line = pd.read_csv(path_line)
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

        return net


    def create_load(path_demand, path_busload, path_bus):
        """
        adapts the load files

        :param path_demand:
        :param path_busload:
        :param path_bus:
        :return: the net with the loads added
        """

        df_demand = pd.read_csv(path_demand)
        df_busload = pd.read_csv(path_busload)
        df_bus = pd.read_csv(path_bus)

        # create basic load dataframe
        # find the bus index of each load
        load_indx = []
        for _, load in df_busload.iterrows():
            bus_load = pp.get_element_index(net, "bus", load.bus)
            load_indx.append(bus_load)

        load_indx = pd.DataFrame(load_indx)
        load_indx = load_indx.rename(columns={0: "bus"})

        # load name and peak power
        load_name = df_busload['bus']
        load_pmw = df_busload['p_mw']
        load_qmvar = df_busload['q_mvar']

        # merge in a full dataframe
        headers = ["name", "bus", "p_mw", "q_mvar"]
        df_load = pd.concat([load_name, load_indx, load_pmw, load_qmvar], axis=1)
        df_load.columns.values[0] = "name"

        # create time series from the basic load df
        Nt = len(df_demand)
        Nl = len(df_load)
        pmw_ts = np.zeros((Nt, Nl), dtype=float)
        for i in range(Nt):  # number of time periods
            pmw_ts[i,:] = df_load['p_mw'][:] * df_demand['norm'][i]

        df_load_ts = pd.DataFrame(pmw_ts, index=range(Nt), columns=load_name.values)
        ds_load_ts = DFData(df_load_ts)
        const_load = control.ConstControl(net, element='load', element_index=load_name.values, variable='p_mw', data_source=ds_load_ts, profile_name=load_name.values)

        return net


    # create empty network
    net = pp.create_empty_network()

    # buses
    net = create_bus(path_bus)

    # lines
    net = create_line(path_line)

    # loads
    net = create_load(path_demand, path_busload, path_bus)
    # df_load = pd.read_csv('Datafiles/demand1.csv')

    # gens

    # trafos



    return net


if __name__ == "__main__":
    path_bus = 'Datafiles/bus1.csv'
    path_line = 'Datafiles/line1.csv'
    path_demand = 'Datafiles/demand1.csv'
    path_busload = 'Datafiles/bus_load1.csv'
    net = initialize_net(path_bus, path_line, path_demand, path_busload)
    print(net.bus)
    print(net.line)

# also add trafos and loads
