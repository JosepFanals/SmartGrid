import pandapower as pp
import pandas as pd

def initialize_net(path_bus, path_line):
    """
    initialize the grid from the .csv files

    :param path_bus: path to the bus .csv file
    :param path_line: path to the line .csv file
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
        pp.create_line(net,
                       from_bus,
                       to_bus,
                       length_km=line.length,
                       std_type=line.std_type,
                       name=line.line_name,
                       parallel=line.parallel)

    return net


if __name__ == "__main__":
    path_bus = 'Datafiles/bus1.csv'
    path_line = 'Datafiles/line1.csv'
    net = initialize_net(path_bus, path_line)
    print(net.bus)
    print(net.line)

# also add trafos and loads
