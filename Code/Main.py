import pandapower as pp
import pandas as pd
import numpy as np
import pandapower.control as control
import pandapower.networks as nw
import pandapower.timeseries as timeseries
import itertools
from pandapower.timeseries.data_sources.frame_data import DFData
from pandapower.plotting import simple_plot

import sys
import codecs

from line_param_calc import calc_line



pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def initialize_net(path_bus, path_geodata, path_line, path_demand, path_busload, path_generation, path_busgen, path_trafo):
    """
    initialize the grid from the .csv files

    :param path_bus: path to the bus .csv file
    :param geodata: path to the geodata .csv file
    :param path_line: path to the line .csv file
    :param path_demand: path to the normalized demand .csv file
    :param busload: path to the bus-load look up table .csv file
    :param path_generation: path to the normalized generation .csv file
    :param busgen: path to the bus-generator look up table .csv file
    :param trafo: path to the trafo .csv file
    :return: the net class
    """

    def create_bus(path_bus, path_geodata):
        """
        adapts the data from the bus file (if needed)

        :param path_bus:
        :param path_geodata:
        :return: the net with the buses added
        """

        df_bus = pd.read_csv(path_bus)
        df_geodata = pd.read_csv(path_geodata)

        net.bus = df_bus

        # adapt geodata
        for ll in range(len(df_geodata)):
            indx_bus = pp.get_element_index(net, "bus", df_geodata['name'][ll])
            df_geodata['name'][ll] = indx_bus

        net.bus_geodata = df_geodata

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
                                         int(line.parallel),
                                         line.Rca,
                                         line.Dext,
                                         line.kg)

            pp.create_line_from_parameters(net,
                                           from_bus,
                                           to_bus,
                                           length_km=line.length,
                                           r_ohm_per_km=rr,
                                           x_ohm_per_km=xx,
                                           c_nf_per_km=cc,
                                           max_i_ka=imax,
                                           name=line.name_l,
                                           parallel=line.parallel,
                                           in_service=line.in_service)

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

        # timeseries
        df_pload_ts = pd.DataFrame(pmw_ts, index=list(range(Nt)), columns=net.load.index)
        df_qload_ts = pd.DataFrame(qmvar_ts, index=list(range(Nt)), columns=net.load.index)
        ds_pload_ts = DFData(df_pload_ts)
        ds_qload_ts = DFData(df_qload_ts)
        const_load = control.ConstControl(net, element='load', element_index=net.load.index, variable='p_mw', data_source=ds_pload_ts, profile_name=net.load.index)
        const_load = control.ConstControl(net, element='load', element_index=net.load.index, variable='q_mvar', data_source=ds_qload_ts, profile_name=net.load.index)  # add the reactive like this?

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

        # gen structure for 1 t
        for ll in range(len(df_busgen)):
            pp.create_gen(net, bus=gen_indx['bus'][ll], p_mw=pmw_ts[0, ll], vm_pu=gen_vpu[ll], name=gen_name[ll], index=int(ll))


        # timeseries
        df_gen_ts = pd.DataFrame(pmw_ts, index=list(range(Nt)), columns=net.gen.index)
        ds_gen_ts = DFData(df_gen_ts)
        const_gen = control.ConstControl(net, element='gen', element_index=net.gen.index, variable='p_mw', data_source=ds_gen_ts, profile_name=net.gen.index)

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


    def create_trafo(path_trafo):
        """
        defines the transformers

        :param path_trafo:
        :return: the net with the transformers added
        """

        df_trafo = pd.read_csv(path_trafo)

        # for trafo in df_trafo:
        for _, trafo in df_trafo.iterrows():
            hv_bus = pp.get_element_index(net, "bus", trafo.hv_bus)
            lv_bus = pp.get_element_index(net, "bus", trafo.lv_bus)

            pp.create_transformer_from_parameters(net,
                                                  hv_bus,
                                                  lv_bus,
                                                  trafo.sn_mva,
                                                  trafo.vn_hv_kv,
                                                  trafo.vn_lv_kv,
                                                  trafo.vkr_percent,
                                                  trafo.vk_percent,
                                                  trafo.pfe_kw,
                                                  trafo.i0_percent,
                                                  in_service=trafo.in_service)

        return net


    # create empty network
    net = pp.create_empty_network()

    # buses
    net = create_bus(path_bus, path_geodata)

    # lines
    net = create_line(path_line)

    # loads
    net = create_load(path_demand, path_busload, path_bus)

    # gens
    net = create_generator(path_generation, path_busgen, path_bus)

    # interconnection
    net = create_intercon(path_bus)

    # trafos
    net = create_trafo(path_trafo)

    return net



def get_Plosses(net, prnt=False):
    """
    returns the active power losses

    :param net: full grid
    :param prnt: to print the value
    :return: total active power losses
    """

    Pll = sum(net.res_line['pl_mw'])
    if prnt is True:
        print('The active power losses are: ', Pll, 'MW')

    return Pll


def get_max_loading(net, prnt=False):
    """
    returns the maximum loading in percentage

    :param net: full grid
    :param prnt: to print the value
    :return: maximum percentual loading of the lines
    """

    L_max = max(net.res_line['loading_percent'])
    if prnt is True:
        print('The maximum loading is: ', L_max, '%')

    return L_max


def run_store_timeseries(net, identifier):
    """
    run the timeseries and store the data

    :param net: full grid
    :param identifier: the name to append to the .xlsx file
    :return: nothing, just store in .xlsx
    """

    ow = timeseries.OutputWriter(net, output_path="./Results/Cases/Case_"+identifier, output_file_type=".xlsx")

    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_line', 'pl_mw')
    ow.log_variable('line', 'parallel')
    ow.log_variable('load', 'p_mw')  # try to read the load to calculate then the efficiency
    # timeseries.run_timeseries(net)
    timeseries.run_timeseries(net, continue_on_divergence=True)

    # store other data, like state of the lines...
    df_lines_states = pd.DataFrame([net.line['name'], net.line['from_bus'], net.line['to_bus'], net.line['in_service']])
    df_lls = df_lines_states.T
    df_lls.to_excel("./Results/Cases/Case_"+identifier+"/lines_states.xlsx")

    # store diagnostic
    diagn = pp.diagnostic(net, report_style='compact')
    # df_diagn = pd.DataFrame(diagn)
    df_diagn = pd.DataFrame.from_dict(list(diagn))
    df_diagn.to_excel("./Results/Cases/Case_"+identifier+"/diagnostic.xlsx")

    return ()


def perms(n_extra_lines):
    """
    generate the permutations

    :param n_extra_lines: number of added lines to combine
    :return: list of permutations
    """
    # lst = ['True', 'False'] * n_extra_lines
    lst = [True, False] * n_extra_lines
    perms_all = set(itertools.permutations(lst, int(n_extra_lines)))
    perms_all = list(perms_all)

    return perms_all


def run_contingencies_ts(path_bus, path_geodata, path_line, path_demand, path_busload, path_generation, path_busgen, path_trafo, n_extra_lines=0):
    """
    run contingencies by disconnecting lines for now

    :param path: path of the datafiles to create the grid
    :param n_extra_lines: number of added lines to combine, the last ones in the .csv
    :return: store in xlsx files
    """
    perms_extra = perms(n_extra_lines)

    net_ini = initialize_net(path_bus, path_geodata, path_line, path_demand, path_busload, path_generation, path_busgen, path_trafo)

    n_cases = len(perms_extra)
    n_lines = len(net_ini.line)

    # disconnect the initial lines and run the cross cases with connecting the others
    for jj in range(n_lines - n_extra_lines):
        net_2 = pp.pandapowerNet(net_ini)
        net_2.line['in_service'][jj] = False

        # evaluate the state by connecting the extra lines
        for kk in range(n_cases):
            # copy the net
            net_3 = pp.pandapowerNet(net_2)

            # change state of the line (True/False)
            net_3.line['in_service'][n_lines - n_extra_lines:] = perms_extra[kk][:]

            # run timeseries and store
            run_store_timeseries(net_3, str(jj) + '_' + str(kk))
            # run_store_timeseries(net_3, str(jj))
            # run_store_timeseries(net_3, str(hash(str(jj) + '_' + str(kk))))

    # return ()
    return n_lines, n_extra_lines, n_cases


def find_optimal_config(path_diagN, path_lineN, path_loadN, path_plN, path_vpuN, path_parallelN, path_pmwN, n_lines, n_extra_lines, n_cases):
    """
    Find the optimal configuration of lines considering convergence, losses, voltages, and costs

    :param path_diagN: path of the diagnostic N cases
    :param path_lineN: path of the line N cases
    :param path_loadN: path of the loading of the lines N cases
    :param path_plN: path of the losses through the lines N cases
    :param path_vpuN: path of the voltages N cases
    :param path_parallelN: path of the number of parallel lines, N cases
    :param path_pmwloadN: path of the MW of load in the buses, N cases

    :return: optimal case
    """

    diag = pd.read_excel(path_diagN, sheet_name=None)
    line = pd.read_excel(path_lineN, sheet_name=None)
    load = pd.read_excel(path_loadN, sheet_name=None)
    pl = pd.read_excel(path_plN, sheet_name=None)
    vpu = pd.read_excel(path_vpuN, sheet_name=None)
    parallel = pd.read_excel(path_parallelN, sheet_name=None)
    pmwload = pd.read_excel(path_pmwN, sheet_name=None)


    on_off_all_lines = []
    for name, sheet in diag.items():
        if len(sheet) == 0:  # if no diagnostic errors
            on_off_lines = line[name]['in_service']
            on_off_all_lines.append(on_off_lines)
            n_parallel = parallel[name]
            n_parallel_subset = n_parallel.loc[0, 0:].T  # only the first row, all are equal

            # loadings
            loadings = load[name]
            loadings_subset = loadings.loc[0:, 0:]
            conditional_loading = loadings_subset[loadings_subset[:] > 80].isnull().values.all()
            # if True, good, we are below 80%

            # vpu
            vpuss = vpu[name]
            vpuss_subset = vpuss.loc[0:, 0:]
            conditional_vpu1 = vpuss_subset[vpuss_subset[:] > 1.10].isnull().values.all()
            conditional_vpu2 = vpuss_subset[0.01 < vpuss_subset[:]][vpuss_subset[:] < 0.90].isnull().values.all()  # to avoid the 0.0 
            # if both are True, we are good

            # active power losses
            pls = pl[name]
            pls_subset = pls.loc[0:, 0:]
            pmws = pmwload[name]
            pmw_subset = pmws.loc[0:, 0:]

            ok_losses = True
            kk = 0
            while ok_losses is True and kk < 24:  # calculate efficiency at any time
                P_load_all = sum(pmw_subset.loc[kk,:])
                P_loss_all = sum(pls_subset.loc[kk,:])
                P_gen_all = P_load_all + P_loss_all
                eff = P_load_all / P_gen_all
                if eff < 0.98: 
                    ok_losses = False
                kk += 1

            # if ok_losses is True, we are good

            full_condition = conditional_loading and conditional_vpu1 and conditional_vpu2 and ok_losses
            if full_condition is True:
                print(name)
        


    return ()






def process_contingencies(n_lines, n_extra_lines, n_cases):
    """
    merge all excels into one but different sheets

    :param n_lines: number of total lines
    :param n_extra_lines: number of added lines
    :param n_cases: resulting total number of cases
    :return: nothing, just store
    """

    dd0 = pd.DataFrame([])

    nnx = n_cases * (n_lines - n_extra_lines)
    f0_vpu = dd0.to_excel("./Results/All_vpu_" + str(nnx) + ".xlsx")
    f0_load = dd0.to_excel("./Results/All_load_" + str(nnx) + ".xlsx")
    f0_pl = dd0.to_excel("./Results/All_pl_" + str(nnx) + ".xlsx")
    f0_diag = dd0.to_excel("./Results/All_diag_" + str(nnx) + ".xlsx")
    f0_line = dd0.to_excel("./Results/All_line_" + str(nnx) + ".xlsx")
    f0_parallel = dd0.to_excel("./Results/All_parallel_" + str(nnx) + ".xlsx") 
    f0_pmw = dd0.to_excel("./Results/All_pmw_" + str(nnx) + ".xlsx")


    w_vpu = pd.ExcelWriter("./Results/All_vpu_" + str(nnx) + ".xlsx")
    w_load = pd.ExcelWriter("./Results/All_load_" + str(nnx) + ".xlsx")
    w_pl = pd.ExcelWriter("./Results/All_pl_" + str(nnx) + ".xlsx")
    w_diag = pd.ExcelWriter("./Results/All_diag_" + str(nnx) + ".xlsx")
    w_line = pd.ExcelWriter("./Results/All_line_" + str(nnx) + ".xlsx")
    w_parallel = pd.ExcelWriter("./Results/All_parallel_" + str(nnx) + ".xlsx")
    w_pmw = pd.ExcelWriter("./Results/All_pmw_" + str(nnx) + ".xlsx")

    for jj in range(n_lines - n_extra_lines):
        for kk in range(n_cases):
            fold_path = "./Results/Cases/Case_" + str(jj) + "_" + str(kk)
            f1_vpu = pd.read_excel(fold_path + "/res_bus/vm_pu.xlsx")
            f1_load = pd.read_excel(fold_path + "/res_line/loading_percent.xlsx")
            f1_pl = pd.read_excel(fold_path + "/res_line/pl_mw.xlsx")
            f1_diag = pd.read_excel(fold_path + "/diagnostic.xlsx")
            f1_line = pd.read_excel(fold_path + "/lines_states.xlsx")
            f1_parallel = pd.read_excel(fold_path + "/line/parallel.xlsx")
            f1_pmw = pd.read_excel(fold_path + "/load/p_mw.xlsx")

            f1_vpu.to_excel(w_vpu, sheet_name=str(jj) + '_' + str(kk))
            f1_load.to_excel(w_load, sheet_name=str(jj) + '_' + str(kk))
            f1_pl.to_excel(w_pl, sheet_name=str(jj) + '_' + str(kk))
            f1_diag.to_excel(w_diag, sheet_name=str(jj) + '_' + str(kk))
            f1_line.to_excel(w_line, sheet_name=str(jj) + '_' + str(kk))
            f1_parallel.to_excel(w_parallel, sheet_name=str(jj) + '_' + str(kk))
            f1_pmw.to_excel(w_pmw, sheet_name=str(jj) + '_' + str(kk))

    w_vpu.save()
    w_load.save()
    w_pl.save()
    w_diag.save()
    w_line.save()
    w_parallel.save()
    w_pmw.save()

    return ()


if __name__ == "__main__":
    # load paths
    path_bus = 'Datafiles/phII/bus1.csv'
    path_geodata = 'Datafiles/phII/geodata1.csv'
    path_line = 'Datafiles/phII/line1.csv'
    path_demand = 'Datafiles/phII/demand1.csv'
    path_busload = 'Datafiles/phII/bus_load1.csv'
    path_generation = 'Datafiles/phII/generation1.csv'
    path_busgen = 'Datafiles/phII/bus_gen1.csv'
    path_trafo = 'Datafiles/phII/trafo1.csv'

    # define net
    net = initialize_net(path_bus, path_geodata, path_line, path_demand, path_busload, path_generation, path_busgen, path_trafo)

    # run and store timeseries data, for only 1 case
    # run_store_timeseries(net, '000')

    # run contingencies
    # n_lines, n_extra_lines, n_cases = run_contingencies_ts(path_bus, path_geodata, path_line, path_demand, path_busload, path_generation, path_busgen, path_trafo, n_extra_lines=6)

    # merge excels
    # process_contingencies(n_lines, n_extra_lines, n_cases)

    path_diagN = 'Results/All_diag_320.xlsx'
    path_lineN = 'Results/All_line_320.xlsx'
    path_loadN = 'Results/All_load_320.xlsx'
    path_plN = 'Results/All_pl_320.xlsx'
    path_vpuN = 'Results/All_vpu_320.xlsx'
    path_parallelN = 'Results/All_parallel_320.xlsx'
    path_pmwN = 'Results/All_pmw_320.xlsx'


    find_optimal_config(path_diagN, path_lineN, path_loadN, path_plN, path_vpuN, path_parallelN, path_pmwN, 11, 6, 64)



    # ------------- Others -------------

    # run diagnostic
    # pp.diagnostic(net)
    # print(net.res_load)
    # print(net.res_line)
    # print(net.res_gen)

    # plot
    # pp.plotting.simple_plot(net)
    # simple_plot(net)

    # get_Plosses(net, prnt=True)
    # get_max_loading(net, prnt=True)


