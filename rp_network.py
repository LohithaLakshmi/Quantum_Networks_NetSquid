""" NetSquid and Python libraries """
import netsquid as ns
import netsquid.qubits.state_sampler as ss
import netsquid.qubits.ketstates as ks
import pandas
import numpy as np
from netsquid.nodes import Node, Network
from netsquid.components.component import *
from netsquid.components.models import FixedDelayModel
from netsquid.components.models import DepolarNoiseModel, DephaseNoiseModel
from netsquid.components.qprogram import QuantumProgram
from netsquid.components import QuantumProcessor, PhysicalInstruction
from netsquid.components import instructions as ins
from netsquid.protocols import NodeProtocol, LocalProtocol
from netsquid.components.instructions import *
from netsquid.components.qprogram import *
from netsquid.examples.teleportation import EntanglingConnection, ClassicalConnection
from netsquid.examples.repeater_chain import *
from netsquid.util import DataCollector
from pydynaa import EventExpression
from netsquid.protocols.protocol import *
from matplotlib import pyplot as plt


""" Repeater Protocol setup with sub protocols SwapProtocol and CorrectProtocol """
def repeater_protocol(network):
    protocol = LocalProtocol(nodes = network.nodes)
    nodes = [network.nodes[name] for name in sorted(network.nodes.keys())]
    for node in nodes[1:-1]:
        sub_protocol = SwapProtocol(node = node, name = f"Swap_{node.name}")
        protocol.add_subprotocol(sub_protocol)
    sub_protocol = CorrectProtocol(nodes[-1], len(nodes))
    protocol.add_subprotocol(sub_protocol)
    
    return protocol

""" QuantumProcessor(QMemory) Initialization """
def qmemory_initialize(name, depolar_rate, dephase_rate):

    physical_instructions = [
        PhysicalInstruction(INSTR_X, duration = 1, quantum_noise_model = DephaseNoiseModel(dephase_rate)),
        PhysicalInstruction(INSTR_Z, duration = 1, quantum_noise_model = DephaseNoiseModel(dephase_rate)),
        PhysicalInstruction(INSTR_MEASURE_BELL, duration = 1),
    ]
    final_qproc = QuantumProcessor(name, num_positions = 2, fallback_to_nonphysical = False,
                                   mem_noise_models = [DepolarNoiseModel(depolar_rate)] * 2, phys_instructions = physical_instructions)
    return final_qproc

""" Repeater Network Setup """
def repeater_network(num_nodes, node_distance, source_frequency, depolar_rate, dephase_rate):
    if num_nodes < 3:
        raise ValueError("The Repeater Network cannot be created with the input nodes")
    network = Network("Repeater_network")
    nodes = []
    for i in range(num_nodes):
        num_zeros = int(np.log10(num_nodes)) + 1
        nodes.append(Node(f"Node_{i:0{num_zeros}d}", qmemory = qmemory_initialize(f"qproc_{i}", depolar_rate = depolar_rate, dephase_rate = dephase_rate)))
    network.add_nodes(nodes)
    for i in range(num_nodes - 1):
        node, node_r = nodes[i], nodes[i + 1]
        quantum_conn = EntanglingConnection(name = f"quantum_conn_{i}-{i+1}", length = node_distance,
                                            source_frequency = source_frequency)
        for q_channel_name in ['qchannel_C2A', 'qchannel_C2B']:
            quantum_conn.subcomponents[q_channel_name].models['quantum_noise_model'] = FibreDepolarizeModel()
        port_name, port_name_r = network.add_connection(node, node_r, connection = quantum_conn, label = "quantum channel connection")
        node.ports[port_name].forward_input(node.qmemory.ports["qin0"])
        node_r.ports[port_name_r].forward_input(node_r.qmemory.ports["qin1"])
        classic_conn = ClassicalConnection(name = f"ccon_{i}-{i+1}", length = node_distance)
        port_name, port_name_r = network.add_connection(node, node_r, connection = classic_conn, label = "classical channel connection",
                                                        port_name_node1 = "ccon_R", port_name_node2 = "ccon_L")
        if "ccon_L" in node.ports:
            node.ports["ccon_L"].bind_input_handler(
                lambda message, _node = node: _node.ports["ccon_R"].tx_output(message))
    return network

""" Fidelity data collection on success event of CorrectProtocol """
def fidelity_datacollector(network, protocol):
    nodes = [network.nodes[name] for name in sorted(network.nodes.keys())]
    
    def fidelity_calculate(evexpr):
        qubit_source, = nodes[0].qmemory.peek([0])
        qubit_destination, = nodes[-1].qmemory.peek([1])
        fidelity = ns.qubits.fidelity([qubit_source, qubit_destination], ks.b00)
        return {"fidelity": fidelity}

    dc = DataCollector(fidelity_calculate, include_entity_name = False)
    dc.collect_on(EventExpression(source = protocol.subprotocols['CorrectProtocol'], event_type = Signals.SUCCESS.value))
    return dc

""" Network Simulation Setup with appropriate source frequency and sim run time """
def network_simulation(num_nodes, node_distance, num_iters, depolar_rate, dephase_rate):
    
    ns.sim_reset()
    estimated_runtime = (0.5 + num_nodes - 1) * node_distance * 5e3
    network = repeater_network(num_nodes = num_nodes, node_distance = node_distance/num_nodes,
                            source_frequency = 1e9 / estimated_runtime, 
                            depolar_rate = depolar_rate,
                            dephase_rate = dephase_rate)
    print("Network Setup done")
    protocol = repeater_protocol(network)
    print("Repeater Protocol Setup done")
    dc = fidelity_datacollector(network, protocol)
    protocol.start()   
    print("Simulation Start")
    ns.sim_run(estimated_runtime * num_iters)
    return dc.dataframe

""" Node vs Fidelity Plot """
def plot_distance(num_iters, depolar_rate, dephase_rate):

    fig, ax = plt.subplots()
    for distance in [10, 20, 30]:
        data = pandas.DataFrame()
        for num_node in range(3, 20):
            data[num_node] = network_simulation(num_nodes = num_node, node_distance = distance/num_node,
                                                num_iters = num_iters,  depolar_rate = depolar_rate, dephase_rate = dephase_rate)['fidelity']
        data = data.agg(['mean', 'sem']).T.rename(columns = {'mean': 'fidelity'})
        data.plot(y = 'fidelity', yerr = 'sem', label = f"Distance: {distance} km", ax = ax)
    plt.xlabel("Nodes")
    plt.ylabel("Fidelity")
    plt.title("Repeater Network for different node distance")
    plt.savefig("rp_ntwk_distance.png")

""" Depolar Rate vs Fidelity Plot """
def plot_depolar(num_iters, num_node, input_distance, dephase_rate):

    depolar_rates = [1e6 * i for i in range(0, 200, 10)]
    data = pandas.DataFrame()
    for depolar_rate in depolar_rates:
        data[depolar_rate] = network_simulation(num_nodes = num_node, node_distance = input_distance/num_node,
                                                num_iters = num_iters, depolar_rate = depolar_rate, dephase_rate = dephase_rate)['fidelity']
    data = data.agg(['mean', 'sem']).T.rename(columns = {'mean': 'fidelity'}).reset_index() 
    plot_style = {'kind': 'scatter', 'grid': True, 'title': "Fidelity of the Repeater Network over different Depolar rates"}
    data.plot(x = 'index' , y = 'fidelity', **plot_style)    
    plt.xlabel("Depolar Rate")
    plt.ylabel("Fidelity")
    plt.savefig("rp_ntwk_depolar.png")
    
""" Dephase Rate vs Fidelity Plot """    
def plot_dephase(num_iters, num_node, input_distance, depolar_rate):

    dephase_rates = [1e6 * i for i in range(0, 200, 10)]
    data = pandas.DataFrame()
    for dephase_rate in dephase_rates:
        data[dephase_rate] = network_simulation(num_nodes = num_node, node_distance = input_distance/num_node,
                                                num_iters = num_iters, depolar_rate = depolar_rate, dephase_rate = dephase_rate)['fidelity']
    data = data.agg(['mean', 'sem']).T.rename(columns = {'mean': 'fidelity'}).reset_index() 
    plot_style = {'kind': 'scatter', 'grid': True, 'title': "Fidelity of the Repeater Network over different Dephase rates"}
    data.plot(x = 'index' , y = 'fidelity', **plot_style)    
    plt.xlabel("Dephase Rate")
    plt.ylabel("Fidelity")
    plt.savefig("rp_ntwk_dephase.png")    


""" Simulation initialization """
run_stats = network_simulation(num_nodes = 16, node_distance = 25, num_iters = 100, depolar_rate = 200, dephase_rate = 200)
print(run_stats)
print("Simulation complete")
plot_distance(num_iters = 1000, depolar_rate = 150, dephase_rate = 150)
plot_depolar(num_iters = 1000, num_node = 16, input_distance = 10, dephase_rate = 150)
plot_dephase(num_iters = 1000, num_node = 16, input_distance = 10, depolar_rate = 150)
