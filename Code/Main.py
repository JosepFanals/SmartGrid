import pandapower as pp
import pandas as pd
import numpy as np
import pandapower.control as control
import pandapower.networks as nw
import pandapower.timeseries as timeseries
from pandapower.timeseries.data_sources.frame_data import DFData
from line_param_calc import calc_line

def initialize_net(path_bus, path_line, path_demand, path_busload, path_generation, path_busgen):
    """
    initialize the grid from the .csv files

    :param path_bus: path to the bus .csv file
    :param path_line: path to the line .csv file
    :param path_demand: path to the normalized demand .csv file
    :param busload: path to the bus-load look up table .csv file
    :param path_generation: path to the normalized generation .csv file
    :param busgen: path to the bus-generator look up table .csv file
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
        qmvar_ts = np.zeros((Nt, Nl), dtype=float)
        for i in range(Nt):  # number of time periods
            pmw_ts[i,:] = df_load['p_mw'][:] * df_demand['norm'][i]
            qmvar_ts[i,:] = df_load['q_mvar'][:] * df_demand['norm'][i]

        # form loads as a static picture (initial time)
        for ll in range(len(df_busload)):
            pp.create_load(net, bus=load_indx['bus'][ll], p_mw=pmw_ts[0, ll], q_mvar=qmvar_ts[0, ll], name=load_name[ll], index=int(ll))

        # add qmvar
        # df_load_ts = pd.DataFrame(pmw_ts, index=list(range(Nt)), columns=load_name.values)

        # df_load_ts = pd.DataFrame(pmw_ts, index=list(range(Nt)), columns=net.load.index)
        # ds_load_ts = DFData(df_load_ts)
        # const_load = control.ConstControl(net, element='load', element_index=net.load.index, variable={'p_mw', 'q_mvar'}, data_source=ds_load_ts, profile_name=net.load.index)

        return net


    def create_generator(path_generation, path_busgen, path_bus):
        """
        adapts the generation files

        :param path_generation:
        :param path_busgenerator:
        :param path_bus:
        :return: the net with the generators added
        """

        df_generation = pd.read_csv(path_generation)
        df_busgen = pd.read_csv(path_busgen)
        df_bus = pd.read_csv(path_bus)

        # create basic generator dataframe
        # find the bus index of each gen
        gen_indx = []
        for _, gen in df_busgen.iterrows():
            bus_gen = pp.get_element_index(net, "bus", gen.bus)
            gen_indx.append(bus_gen)

        gen_indx = pd.DataFrame(gen_indx)
        gen_indx = gen_indx.rename(columns={0: "bus"})

        # load name and peak power
        gen_name = df_busgen['bus']
        gen_pmw = df_busgen['p_mw']
        gen_vpu = df_busgen['vm_pu']

        # merge in a full dataframe
        headers = ["name", "bus", "p_mw", "vm_pu"]
        df_gen = pd.concat([gen_name, gen_indx, gen_pmw, gen_vpu], axis=1)
        df_gen.columns.values[0] = "name"

        # create time series from the basic load df
        Nt = len(df_generation)
        Ng = len(df_gen)
        pmw_ts = np.zeros((Nt, Ng), dtype=float)
        for i in range(Nt):  # number of time periods
            pmw_ts[i,:] = df_gen['p_mw'][:] * df_generation['norm'][i]


        for ll in range(len(df_busgen)):
            pp.create_gen(net, bus=gen_indx['bus'][ll], p_mw=pmw_ts[0, ll], vm_pu=gen_vpu[ll], name=gen_name[ll], index=int(ll))


        # df_gen_ts = pd.DataFrame(pmw_ts, index=list(range(Nt)), columns=gen_name.values)
        # ds_gen_ts = DFData(df_gen_ts)
        # const_gen = control.ConstControl(net, element='gen', element_index=gen_name.values, variable='p_mw', data_source=ds_gen_ts, profile_name=gen_name.values)

        return net


    def create_intercon(path_bus):
        """
        defines the interconnection (slack bus)

        :param path_bus:
        :return: the net with the interconnection added
        """

        df_bus = pd.read_csv(path_bus)

        # find the slack index
        slack_indx = 0
        for ll in range(len(df_bus)):
            # slack_indx = pp.get_element_index(net, "bus", bb.name)
            if df_bus['name'][ll] == 'intercon':
                slack_indx = pp.get_element_index(net, "bus", df_bus['name'][ll])

        pp.create_ext_grid(net, slack_indx, vm_pu=1.0, va_degree=0)

        return net



    # create empty network
    net = pp.create_empty_network()

    # buses
    net = create_bus(path_bus)

    # lines
    net = create_line(path_line)

    # loads
    net = create_load(path_demand, path_busload, path_bus)

    # gens
    net = create_generator(path_generation, path_busgen, path_bus)

    # interconnection
    net = create_intercon(path_bus)

    # trafos



    return net


if __name__ == "__main__":
    path_bus = 'Datafiles/bus1.csv'
    path_line = 'Datafiles/line1.csv'
    path_demand = 'Datafiles/demand1.csv'
    path_busload = 'Datafiles/bus_load1.csv'
    path_generation = 'Datafiles/generation1.csv'
    path_busgen = 'Datafiles/bus_gen1.csv'

    net = initialize_net(path_bus, path_line, path_demand, path_busload, path_generation, path_busgen)

    # pp.runpp(net)
    pp.diagnostic(net)
    # timeseries.run_timeseries(net)

# also add trafos and loads
