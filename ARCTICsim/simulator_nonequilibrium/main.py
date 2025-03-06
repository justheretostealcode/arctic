"""
Author: Erik Kubaczka
The code of Nicolai Engelmann from circuit_simulator_thermo.py was adapted to fit the needs of the new simulator.
"""
import cProfile
import json
import os
import inspect
import os.path as op
import shlex
from argparse import ArgumentParser

from simulator.circuit import GeneticLogicCircuit
from simulator.circuit_evaluator_new import CircuitEvaluator

from simulator.circuit_utils import CircuitAssignment, load_structure
from simulator.gatelib import GateLibCollectionBased
from simulator.utils import JsonFile, communication_wrapper, load_settings, type_dict
from utils.utils import CustomJsonEncoder

here = op.dirname(op.abspath(inspect.getfile(inspect.currentframe())))
version = '0.9'
settings_config_path = './settings_config.cfg'


def sim_run(lineargs, json_str=None):
    global evaluator, argp, settings
    args = " ".join(lineargs)
    sim_settings = {}
    sim_settings.update(settings)
    sim_settings.pop("structure")
    sim_settings_update = {k: v for k, v in vars(argp.parse_args(shlex.split(args, posix=False))).items()
                           if v is not None}
    sim_settings.update(sim_settings_update)
    assignment_string = sim_settings["assignment"]

    #

    assignment_string = assignment_string.replace("\'", "")
    json_assignment = None
    if os.path.exists(assignment_string):
        json_assignment = JsonFile(path=assignment_string)
    else:
        json_assignment = JsonFile(content=assignment_string)
    assignment = CircuitAssignment(json_file=json_assignment, gate_lib=evaluator.gate_lib)

    old_structure = evaluator.structure
    if "structure" in sim_settings:
        structure = load_structure(sim_settings)
        evaluator.set_structure(structure)

    # profiler = cProfile.Profile()
    # profiler.enable()

    result = evaluator.score_assignment(assignment=assignment, sim_settings=sim_settings)

    # profiler.disable()
    # profiler.dump_stats("Simulation_Profile.prof")
    # profiler.print_stats()

    if "structure" in sim_settings:
        evaluator.set_structure(old_structure)

    return result


# start -a {"a":1}
# start -a={"a":"input_3","b":"input_1","c":"input_2","NOT_0":"P1_PsrA","NOT_2":"P1_IcaR","NOT_4":"P1_PhlF","NOR2_1":"P1_QacR","NOR2_3":"P1_HKCI","O":"output_1"}
# start -a={"a":"input_3","b":"input_1","c":"input_2","NOT_0":"P1_PhlF","NOT_2":"P1_IcaR","NOT_4":"P1_PsrA","NOR2_1":"P1_QacR","NOR2_3":"P1_HKCI","O":"output_1"}

def sim_plasmid(lineargs):
    global evaluator, argp, settings
    args = " ".join(lineargs)
    sim_settings = {}
    sim_settings.update(settings)
    sim_settings.pop("structure")
    sim_settings_update = {k: v for k, v in vars(argp.parse_args(shlex.split(args, posix=False))).items() if
                           v is not None}
    sim_settings.update(sim_settings_update)
    assignment_string = sim_settings["assignment"]

    assignment_string = assignment_string.replace("\'", "")
    json_assignment = None
    if os.path.exists(assignment_string):
        json_assignment = JsonFile(path=assignment_string)
    else:
        json_assignment = JsonFile(content=assignment_string)
    assignment = CircuitAssignment(json_file=json_assignment, gate_lib=evaluator.gate_lib)

    old_structure = evaluator.structure
    if "structure" in sim_settings:
        structure = load_structure(sim_settings)
        evaluator.set_structure(structure)

    # profiler = cProfile.Profile()
    # profiler.enable()
    circuit = evaluator.circuit
    circuit.set_assignment(assignment)

    plasmid = [None] * 2
    genes = circuit.genes
    n_genes_per_plasmid = 8
    n_plasmids = 2

    # for iG in range(n_genes_per_plasmid):
    #     for iP in range(n_plasmids):
    for iE, gene_id in enumerate(genes):
        iP = iE % n_plasmids
        iG = iE // 2
        if plasmid[iP] is None:
            plasmid[iP] = [None] * n_genes_per_plasmid

        plasmid[iP][iG] = str(genes[gene_id])
    # profiler.disable()
    # profiler.dump_stats("Simulation_Profile.prof")
    # profiler.print_stats()

    if "structure" in sim_settings:
        evaluator.set_structure(old_structure)

    return plasmid

def sim_exit(lineargs):
    # Finish the simulator
    exit(0)


def sim_update_settings(lineargs):
    global settings, argp
    settings_update = {k: v for k, v in vars(argp.parse_args(shlex.split(lineargs, posix=False))).items()
                       if v is not None}

    settings.update(settings_update)

    if "structure" in settings_update:
        structure = load_structure(settings_update)
        evaluator.set_structure(structure)

    pass


def sim_print_settings(lineargs):
    global settings
    print(settings)
    pass


def sim_print_commands(lineargs):
    raise Exception("Not implemented yet")
    pass


# to add a command to the simulator, extend this list and add your function
# any function *can* accept a string as an argument containing the further
# part of the command line
command_dict = dict({
    'start': [sim_run, 'starts the simulation'],
    'exit': [sim_exit, 'terminates the simulator'],
    'update_settings': [sim_update_settings, 'update the simulation settings'],
    'print_settings': [sim_print_settings, 'print the current simulation settings'],
    'help': [sim_print_commands, 'prints this help'],
    'plasmid': [sim_plasmid, 'Outputs the plasmid information']
})

if __name__ == "__main__":

    # change working directory to here
    os.chdir(here)

    # load default settings first
    settings, types, comments = load_settings(settings_config_path)

    # update settings through the command line
    argp = ArgumentParser(description='Thermodynamic Non-Equilibrium Steady State circuit simulator v.' + version)
    for k, v in settings.items():
        argp.add_argument('-' + k[0], '--' + k, required=False, type=type_dict[types[k]], help=comments[k])

    # The options are not supported
    # argp.add_argument('--autostart', required=False, action='store_true', default=False, help='if set, simulation starts immediately (no CLI beforehand)')
    # argp.add_argument('--autoexit', required=False, action='store_true', default=False, help='if set, termination immediately after first simulation (no CLI afterwards)')
    # argp.add_argument('--no_cli', required=False, action='store_true', default=False, help='if set, no CLI is called at all (immediate start, immediate exit)')
    settings.update({k: v for k, v in vars(argp.parse_args()).items() if v is not None})

    # setup communication interface
    cli_io = communication_wrapper(None, None, prefix=':>')

    path_to_library = settings["library"]
    json_lib = JsonFile(path=path_to_library)

    gate_lib = GateLibCollectionBased(json_file=json_lib)

    structure = None
    if "structure" in settings:
        structure = load_structure(settings)

    # Interface to new simulator
    evaluator = CircuitEvaluator(gate_lib=gate_lib, settings=settings, structure=structure)

    cli_io.writeline("ready")

    # run CLI loop
    while True:
        # read in the next line and split it into command and arguments
        line_cmd, *line_args = tuple(map(str.strip, cli_io.readline().split(' ', 1)))

        func = command_dict[line_cmd][0]

        result = func(line_args)
        json_content = json.dumps(result, cls=CustomJsonEncoder)
        cli_io.writeline(json_content)
