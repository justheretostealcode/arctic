# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 13:52:04 2020

@author: Nicolai Engelmann
"""

import numpy as np
import csv
import math
import matplotlib.pyplot as plt
import json
import time
import os
import sys
from copy import copy, deepcopy


# The interfacing in the simulator methods and marshalling and demarshalling
# of objects is based on json. Therefor, all classes can be generated from
# files or strings handed over via command line. To make the objects writable
# they must implement a __str__() method for marshalling that converts them
# to a json representation, which may then be stored in a file.

# List of content
#     - technology mapping interface related
#         - classes:
#             - simulator_settings
#             - runtime_interface
#     - circuit related classes and functions
#         - classes:
#             - nand_circuit
#             - nor_circuit
#             - circuit_structure
#             - circuit_assignment
#             - library
#         - functions:

# ____________________________________________________________________________
#   Globals and general purpose short-cuts
# ____________________________________________________________________________


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def hl(word):
    return bcolors.WARNING + str(word) + bcolors.ENDC


def head(word):
    return bcolors.OKCYAN + str(word) + bcolors.ENDC


DEBUG_LEVEL = 0


# ____________________________________________________________________________
#   Technology mapping interface related classes
# ____________________________________________________________________________

# class simulator_settings(dict):
#     def __init__(self, *arg, **kw):
#       super(simulator_settings, self).__init__(*arg, **kw)

class execution_wrapper:
    def __init__(self, command_dict):
        self.main = command_dict
        self.pre = dict({k: (lambda *args, **kwargs: (args, kwargs)) for k, v in self.main.items()})
        self.post = dict({k: (lambda ret: ret) for k, v in self.main.items()})
        self.compose = dict({k: [True, True] for k in self.main.keys()})

    def register_pre(self, key, key_or_fun, compose=True):
        if isinstance(key_or_fun, str) and key_or_fun in self.main.keys():
            self.pre[key] = self.main[key_or_fun]
        elif callable(key_or_fun):
            self.pre[key] = fun
        else:
            raise ValueError('Supplied argument is neither callable nor a registered function.')
        self.compose[key][0] = compose

    def register_post(self, key, key_or_fun, compose=True):
        if isinstance(key_or_fun, str) and key_or_fun in self.main.keys():
            self.post[key] = self.main[key_or_fun]
        elif callable(key_or_fun):
            self.post[key] = fun
        else:
            raise ValueError('Supplied argument is neither callable nor a registered function.')
        self.compose[key][1] = compose

    def call(self, key, *args, **kwargs):
        if key not in self.main:
            raise ValueError('Function name to be called unknown.')
        if self.compose[key][0]:
            main_args, main_kwargs = self.pre[key](*args, **kwargs)
            ret = self.main[key](*main_args, **main_kwargs)
        else:
            self.pre[key]()
            ret = self.main[key](*args, **kwargs)
        if self.compose[key][1]:
            return self.post[key](ret)
        return self.post[key]()


class communication_wrapper:
    def __init__(self, input_file, output_file, prefix=None):
        self.i = None
        self.o = None
        self.prefix = prefix
        if input_file is None:
            self.i = sys.stdin
        else:
            pass  # stored and temporary files not supported yet
        if output_file is None:
            self.o = sys.stdout
        else:
            pass  # stored and temporary files not supported yet

    def readline(self):
        # first signal, that you are ready, by printing the prefix
        if self.prefix is not None:
            self.o.write(self.prefix + ' ')
            self.o.flush()
        return self.i.readline()

    def writeline(self, line):
        self.o.write(line + '\n')


# ____________________________________________________________________________
#   Circuit related classes and functions
# ____________________________________________________________________________

# input encoding dict
_input_encoding = dict({
    'a': 0,
    'b': 1,
    'c': 2
})


# A structure for a NOR circuit
# we try it here with a fixed TF-order and all information
class nor_circuit:
    def __init__(self, structure, assignment, library, solver=None):
        self.structure = structure
        self.assignment = assignment
        self.lib = library
        self.gates = None

        # create a total order over all TF's and promoters
        # p_idx creates a dict of indices associated with the specific promoters
        # tf_idx creates a dict of index tuples associated with the specific tf's
        self.p_idx = dict(zip(self.lib.parts['promoters'].keys(), list(range(len(self.lib.parts['promoters'])))))
        self.tf_idx = dict({k: tuple() for k in self.lib.parts['tfs'].keys()})
        self.dev_idx = dict({k: None for k in self.lib.devices.keys()})
        for dev, parts in self.lib.devices.items():
            for tf in parts['tfs']:
                self.tf_idx[tf] += tuple(self.p_idx[p] for p in parts['promoters'])
                # print('tf idx for ' + tf + ': ' + str(self.tf_idx[tf]))
            self.dev_idx[dev] = tuple(self.p_idx[p] for p in parts['promoters'])
            # print('device idx for ' + dev + ': ' + str(self.dev_idx[dev]))

        # get factors for all possible promoters
        self.factors = np.zeros([2, len(self.p_idx), len(self.p_idx)])
        self.affinities = np.zeros(len(self.p_idx))
        self.extremes = np.zeros([len(self.p_idx), 2])
        for p, d in self.lib.parts['promoters'].items():
            self.extremes[self.p_idx[p], 1] = float(d['typical']['on'])
            self.extremes[self.p_idx[p], 0] = float(d['typical']['off'])
            self.affinities[self.p_idx[p]] = float(d['f']['rnap'])
            for tf in self.tf_idx.keys():
                # print(tf + ' at ' + str(self.tf_idx[tf][0]) + ' associated with ' + p + ' at ' + str(self.p_idx[p]))
                for pix in self.tf_idx[tf]:
                    self.factors[0, self.p_idx[p], pix] = float(d['f']['rnap']) * float(d['f']['tf_rnap'][tf])
                    self.factors[1, self.p_idx[p], pix] = float(d['f']['tf_only'][tf])

        # create now a total order of all locally available gates
        node_list = list(self.structure.nodes)
        self.node_idx = dict(zip(node_list, list(range(len(node_list)))))

        # create an array which indicates which gate outputs are mutable
        self.mutable = np.ones(len(node_list))
        for gate in self.structure.inputs:
            self.mutable[self.node_idx[gate]] = 0

        # execute the assignment
        # get the maps g_p, the inverse p_g and w == g_g resolving the
        # circuit transformed mapping p_p == g_p ( w ( p_g ) )
        self.g_p = np.zeros(len(node_list), dtype=int)
        self.p_g = -np.ones(len(self.p_idx), dtype=int)
        self.dummy_idx = np.zeros(len(self.assignment.dummys), dtype=int)
        self.gates = np.array([None for _ in range(len(node_list))])  # array of objects
        for k, v in self.assignment.map_gtod.items():
            # print('node/device: ' + hl(str(k)) + '/' + hl(str(v)) + ', node idx = ' + hl(str(self.node_idx[k])) + ', device idx = ' + hl(str(self.dev_idx[v][0])))
            if v[0] == '_':  # dummy gate
                self.g_p[self.node_idx[k]] = -1
                self.dummy_idx[int(v[1])] = self.node_idx[k]
                # translate the dict describing the dummys behaviour into index notation
                # gate_idx -> env -> a_out, p_idx
                codebook = np.zeros([len(node_list), 2, 2])
                for l, w in self.assignment.dummys[v].items():
                    # TODO: this can be shorter
                    nidx = self.node_idx[l]
                    codebook[nidx, 0, 0] = float(w['a'][1])
                    codebook[nidx, 1, 0] = float(w['a'][0])
                    codebook[nidx, 0, 1] = self.tf_idx[w['b'][1]][0]
                    codebook[nidx, 1, 1] = self.tf_idx[w['b'][0]][0]
                self.gates[self.node_idx[k]] = _dummy_gate(k, v, codebook)
            elif v.startswith('output'):  # implicit or / reporter TF
                self.g_p[self.node_idx[k]] = self.dev_idx[v][0]
                self.p_g[self.dev_idx[v][0]] = self.node_idx[k]
                or_factors = np.zeros(len(self.p_idx))
                or_factors[self.dev_idx[v][0]] = 1.0
                self.gates[self.node_idx[k]] = _implicit_or_gate(k, v, or_factors)
            else:
                self.g_p[self.node_idx[k]] = self.dev_idx[v][0]
                self.p_g[self.dev_idx[v][0]] = self.node_idx[k]
                # Find better solution to recognize YFP
                self.gates[self.node_idx[k]] = _nor_gate(k, v, self.factors[:, self.dev_idx[v][0], :],
                                                         self.affinities[self.dev_idx[v][0]],
                                                         self.extremes[self.dev_idx[v][0], :],
                                                         self.lib.env['reservoir'])
        # also always create the artificial output gates (which are not in assignment)
        # for k in list(self.structure.outputs):
        #    self.g_p[self.node_idx[k]] = -1
        #    self.gates[self.node_idx[k]] = _implicit_or_gate(k, 'simulation_monitor')
        # print all gates if debug level is above 0
        if DEBUG_LEVEL > 0:
            print('Gates comprising the circuit:')
            for gate in self.gates:
                print(hl(str(gate.node)) + '/' + hl(str(gate.dev)) + ', type = ' + str(gate.type))

        # generate the adjacency matrix w == g_g
        if DEBUG_LEVEL > 0:
            print('Drawing wires:')
        self.w = np.zeros([len(node_list), len(node_list)])
        for gate_in, v in self.structure.adjacency['in'].items():
            t = self.node_idx[gate_in]
            for gate_out in v:
                f = self.node_idx[gate_out]
                if DEBUG_LEVEL > 0:
                    print(hl(gate_out) + ' -> ' + hl(gate_in))
                self.w[f, t] = 1

        # for bounding, get a vector of extreme values
        self.bound_a_min = np.array([gate.min for gate in self.gates])
        self.bound_a_max = np.array([gate.max for gate in self.gates])
        self.bound_env = np.zeros(len(self.gates), dtype=int)

        # Now finally set the solver
        if solver is not None:
            self.set_solver(solver)

    def propagate(self, a):
        new_a = np.copy(a)
        wa = np.dot(self.w.T, a)
        p_a = np.zeros(len(self.p_idx))
        # print([(gate.node, self.node_idx[gate.node]) for gate in self.gates])
        for n in range(len(self.gates)):
            if self.g_p[n] != -1:
                p_a[self.g_p[n]] += wa[n]
        for n in range(len(self.gates)):
            if not self.mutable[n]:
                continue
            gate = self.gates[n]
            if DEBUG_LEVEL > 2:
                print('p_a vector for ' + hl(str(gate.node)) + ' of type ' + hl(str(gate.type)) + ':')
                print(p_a)
            new_a[n] = self.gates[n].out(p_a)
        return new_a

    # this will be sloooooow.....
    def propagate_bound(self, a, heuristic=False):
        new_a = np.copy(a)
        if DEBUG_LEVEL > 1:
            print('')
        for n in range(len(self.gates)):
            if not self.mutable[n] or self.gates[n].type == -1:
                if DEBUG_LEVEL > 1:
                    print('Skipping immutable or dummy node: ' + hl(self.gates[n].node))
                continue
            if DEBUG_LEVEL > 1:
                print('Looking at node: ' + hl(self.gates[n].node))
            if DEBUG_LEVEL > 2:
                print(
                    'Propagate: ' + hl(str(self.gates[n].node)) + '/' + hl(str(self.gates[n].dev)) + ', type = ' + str(
                        self.gates[n].type) + ', env = ' + hl(str(self.bound_env[n])))
            p_a = np.zeros(len(self.p_idx))
            # get extreme inputs
            a_x = np.zeros(len(self.gates))
            if self.bound_env[n] == 1:  # everything on minimum!
                a_x = np.copy(self.bound_a_min)
            else:  # everything on maximum!
                a_x = np.copy(self.bound_a_max)
            if DEBUG_LEVEL > 2:
                print('dummy wires:')
            for idx in self.dummy_idx:  # set the dummy gates optimally
                a_x[idx] = self.gates[idx].out(n, self.bound_env[n])
                if DEBUG_LEVEL > 2:
                    print(':: ' + hl(str(a_x[idx])))
            # do the wiring
            wa = np.dot(self.w.T, a_x)
            # but unset the own wire because we reset it later
            wa[n] = 0
            # set the correct wire input
            for m in range(len(self.gates)):
                if self.w[m, n] == 1:  # we have a an input, so set
                    if self.gates[m].type == -1:  # dummy
                        wa[n] += self.gates[m].out(n, self.bound_env[n])
                    else:
                        wa[n] += a[m]
            # now associate the correct TF's
            if DEBUG_LEVEL > 2:
                print('dummy TF\'s:')
            force_env = False
            in_env = np.sum(self.bound_env[np.nonzero(self.w[:, n])])
            if in_env != np.sum(self.w[:, n]) and in_env != 0 and not heuristic:
                force_env = True
            for m in range(len(self.gates)):
                if self.gates[m].type == -1 and self.w[m, n] == 0:  # dummy: choose weakest or strongest TF
                    pa_idx = self.gates[m].promoter_entry(n, self.bound_env[n])
                    if DEBUG_LEVEL > 2:
                        print(':: ' + hl(pa_idx))
                    p_a[pa_idx] += wa[m]
                elif self.gates[n].type != 1 or self.w[m, n] == 1:  # only accept crosstalk if not implicit OR
                    if self.w[m, n] == 1:
                        val = 0
                        forced = False
                        if force_env and self.gates[m].type == 0 and (
                                (self.gates[n].type == 0 and self.bound_env[m] == self.bound_env[n]) or (
                                self.gates[n].type == 1 and self.bound_env[m] != self.bound_env[n])):
                            forced_value = 0
                            if (self.bound_env[n] == self.gates[
                                n].type):  # either NOR gate and 0-env or implicit OR gate and 1-env
                                forced_value = self.bound_a_max[m]
                            else:
                                forced_value = self.bound_a_min[m]
                            val = forced_value
                            forced = True
                        else:
                            if self.gates[m].type != -1:
                                val = a[m]
                            else:
                                val = self.gates[m].out(n, self.bound_env[n])
                        if DEBUG_LEVEL > 1:
                            if forced:
                                print(hl(str(self.gates[m].node)) + ' -> ' + hl(
                                    str(self.gates[n].node)) + ' by wire with ' + head('forced') + ' value ' + hl(
                                    str(val)))
                            else:
                                print(hl(str(self.gates[m].node)) + ' -> ' + hl(
                                    str(self.gates[n].node)) + ' by wire with value ' + hl(str(val)))
                        p_a[self.g_p[n]] += val
                    elif m != n:
                        if DEBUG_LEVEL > 1:
                            print(hl(str(self.gates[m].node)) + ' -> ' + hl(
                                str(self.gates[n].node)) + ' by crosstalk with value ' + hl(str(wa[m])))
                        p_a[self.g_p[m]] += wa[m]
            # propagate the artificial environment through the gate
            if DEBUG_LEVEL > 2:
                print('p_a vector for ' + hl(str(self.gates[n].node)) + ' of type ' + hl(str(self.gates[n].type)) + ':')
                print(p_a)
            new_a[n] = self.gates[n].out(p_a)
        return new_a

    def set_dummy_mode(self, ttix=0):
        # this mode must be set at least once if dummy gates are present
        for n in range(len(self.gates)):
            if self.gates[n].type != -1:  # no dummy
                self.bound_env[n] = self.structure.gate_truthtables[self.gates[n].node][ttix]

    def set_initial_value(self, values, ttix=-1):
        val = np.zeros(len(self.node_idx))
        for input in self.structure.inputs:
            val[self.node_idx[input]] = values[_input_encoding[input]]
        self.solver.set_initial_value(val)
        if ttix != -1:
            self.set_dummy_mode(ttix)

    def set_solver(self, solver_class):
        self.solver = solver_class(self, None, False)

    def solve(self, tol, max_iter):
        if self.solver is not None:
            (values, err, iter) = self.solver.solve(tol, max_iter)
            # currently supports only one output
            return (values[self.node_idx[list(self.structure.outputs)[0]]], err, iter)
        return None

    def is_valid(self):
        pass

    def __len__(self):
        return len(self.node_idx)

    def __str__(self):
        s = "\nWiring:\n" + str(self.w)
        s += "\nGates: " + str([gate.name for gate in self.gates])
        for g in self.gates:
            s += '\n'
            s += 'Gate: ' + str(g)
        return s


class circuit_structure:
    def __init__(self, str):
        circ = json.loads(str)
        if circ is not None:
            self.order = None
            self.truthtable = np.array(list(map(int, circ['truthtable'])))
            self.inputs = set()
            self.gates = set()
            self.outputs = set()
            self.nodes = set()
            self.stats = {'n_inputs': 0, 'n_nodes': 0, 'n_outputs': 0}
            for n in circ['graph']['nodes']:
                if n['type'].startswith('INPUT'):
                    self.stats['n_inputs'] += 1
                    self.inputs.add(n['id'])
                elif n['type'].startswith('OUTPUT'):
                    self.stats['n_outputs'] += 1
                    self.outputs.add(n['id'])
                else:
                    self.stats['n_nodes'] += 1
                    self.gates.add(n['id'])
            self.nodes = self.inputs | self.gates | self.outputs
            self.gate_truthtables = dict()
            # for k, v in circ['gate_truthtables'].items():
            #     self.gate_truthtables[k] = np.flip(np.array(list(map(int, v))))
            self.internal_edges = set()
            self.outgoing_edges = set()
            self.adjacency = {'in': dict_from_list(self.nodes, set()), 'out': dict_from_list(self.nodes, set())}
            for e in circ['graph']['edges']:
                if e['source'] in self.gates and e['target'] in self.gates:
                    self.internal_edges.add(tuple((e['source'], e['target'])))
                else:
                    self.outgoing_edges.add(tuple((e['source'], e['target'])))
                self.adjacency['in'][e['target']].add(e['source'])
                self.adjacency['out'][e['source']].add(e['target'])
                # if e['source'] in self.inputs:
                #     self.adjacency['in'][e['target']].add(e['source'])
                # elif e['target'] in self.outputs:
                #     self.adjacency['out'][e['source']].add(e['target'])
            self.edges = set.union(self.internal_edges, self.outgoing_edges)
            # self.dprint('\n\t+ Circuit has truthtable {f}, {u} inputs, {y} outputs, {n} intermediate nodes as well as {e} internal and {o} outgoing edges.'.format(f=hl(self.truthtable), u=hl(self.stats['n_inputs']), y=hl(self.stats['n_outputs']), n=hl(self.stats['n_nodes']), e=hl(len(self.internal_edges)), o=hl(len(self.outgoing_edges))))
            self.valid = True

    def combinational_order(self):
        if self.order is None:
            self.order = list()
            for gate in self.gates:
                for n in range(len(self.order)):
                    if self.order[n] in self.adjacency['out'][gate]:
                        self.order.insert(n, gate)
        return self.order

    # return the JSON representation of this graph
    def __str__(self):
        pass


# the assignment connects the library with the structure. It therefore needs to know both to sanity check
class circuit_assignment:
    def __init__(self, jstr):
        assi = json.loads(jstr)
        self.map_gtod = dict()
        self.map_dtog = dict()
        self.dummys = dict()
        n_dummy = 0
        for k, v in assi.items():
            d = v
            if isinstance(v, dict):
                # if there is a dict present, we want a dummy gate which emits extreme values
                d = '_' + str(n_dummy)
                self.dummys[d] = v
                n_dummy += 1
            self.map_gtod[k] = d
            self.map_dtog[d] = k
        self.devices = set(self.map_dtog.keys())
        self.gates = set(self.map_dtog.values())


class library:
    # represents a connected gate lib
    def __init__(self, jstr):
        self.parts = dict()
        self.parts['promoters'] = dict()
        self.parts['tfs'] = dict()
        self.devices = dict()
        self.env = dict()
        libstr = json.loads(jstr)
        for entry in libstr:
            if entry['class'] == 'transcription_factors':
                for part in entry['members']:
                    self.parts['tfs'][part['name']] = dict()
                    self.parts['tfs'][part['name']]['part_of'] = part['associated_devices']
                    for dev in part['associated_devices']:
                        if dev not in self.devices:
                            self.devices[dev] = dict()
                            self.devices[dev]['tfs'] = list()
                            self.devices[dev]['promoters'] = list()
                        self.devices[dev]['tfs'] += [part['name']]
                    self.parts['tfs'][part['name']]['typical'] = [part['levels']['off'], part['levels']['on']]
            if entry['class'] == 'promoters':
                for part in entry['members']:
                    self.parts['promoters'][part['name']] = dict()
                    self.parts['promoters'][part['name']]['part_of'] = part['associated_devices']
                    for dev in part['associated_devices']:
                        self.devices[dev]['promoters'] += [part['name']]
                    if 'levels' in part:
                        self.parts['promoters'][part['name']]['typical'] = part['levels']
                    self.parts['promoters'][part['name']]['e'] = part['energies']
                    self.parts['promoters'][part['name']]['f'] = part['factors']
            if entry['class'] == 'environment':
                self.env['name'] = entry['name']
                self.env['beta'] = entry['beta']
                self.env['reservoir'] = entry['non_specific_reservoir']
                self.env['df'] = entry['env_tfs']
                self.env['e'] = entry['reservoir_energies']

    # return the JSON representation of this lib
    def __str__(self):
        content = list()
        content += [dict()]
        content[0]['class'] = 'transcription_factors'
        content[0]['members'] = list()
        for k, v in self.parts['tfs'].items():
            entry = dict()
            entry['name'] = k
            entry['associated_devices'] = v['part_of']
            entry['levels'] = dict()
            entry['levels']['off'] = v['typical'][0]
            entry['levels']['on'] = v['typical'][1]
            content[0]['members'] += [entry]
        content += [dict()]
        content[1]['class'] = 'promoters'
        content[1]['members'] = list()
        for k, v in self.parts['promoters'].items():
            entry = dict()
            entry['name'] = k
            entry['associated_devices'] = v['part_of']
            entry['energies'] = v['energies']
            content[1]['members'] += [entry]
        content += [dict()]
        content[2]['class'] = 'environment'
        content[2]['name'] = self.env['name']
        content[2]['beta'] = self.env['beta']
        content[2]['non_specific_reservoir'] = self.env['reservoir']
        content[2]['env_tfs'] = self.env['df']
        content[2]['reservoir_energies'] = self.env['e']
        return json.dumps(content)


# PRIVATE. Not intended to use manually
# The NOR gate is a simple, modular gate structure independent of input wiring
class _nor_gate:
    # represents a NOR gate in the circuit
    def __init__(self, node, dev, factors, affinity, extreme, c):
        self.node = node
        self.dev = dev
        self.min = extreme[0]
        self.max = extreme[1]
        self.bepj = factors[0]
        self.bef = factors[1]
        self.bep = affinity
        # self.dim = np.shape(factors[0])[1]
        self.c = c
        self.type = 0

    def out(self, wa):
        # iwa = np.cumprod(np.repeat(wa[:, np.newaxis], self.dim, axis=1), axis=1)
        iwa = wa
        if DEBUG_LEVEL > 2:
            print(str(self.c * self.bep + np.sum(iwa * self.bepj)) + '/' + str(
                self.c + np.sum(iwa * self.bef)) + ' = ' + hl(
                str((self.c + np.sum(iwa * self.bepj)) / (self.c + np.sum(iwa * self.bef)))))
            if self.node == 'NOR2_2':
                print(self.bepj)
                print(self.bef)
        return (self.c * self.bep + np.sum(iwa * self.bepj)) / (self.c + np.sum(iwa * self.bef))

    def __str__(self):
        return self.name + ': ' + str(self.e)


# PRIVATE. Not intended to use manually
# The implicit OR gate is just a pseudo-gate, which outputs the YFP RPU
class _implicit_or_gate:
    # represents a NOR gate in the circuit
    def __init__(self, node, dev, factors):
        self.node = node
        self.dev = dev
        self.type = 1
        self.min = 0
        self.max = 0
        self.b = factors.astype(bool)

    def out(self, wa):
        return np.sum(wa[self.b])

    def __str__(self):
        return self.name + ': None (implicit OR)'


# PRIVATE. Not intended to use manually
# The dummy gate is a placeholder, which always emits 0 but has initialised
# boundary values
class _dummy_gate:
    # represents a NOR gate in the circuit
    def __init__(self, node, dev, codebook):
        self.node = node
        self.dev = dev
        self.codebook = codebook
        self.type = -1
        self.min = 0
        self.max = 0

    def out(self, g_idx, env):
        return self.codebook[g_idx, env, 0]

    def promoter(self, g_idx, env):
        return int(self.codebook[g_idx, env, 1])

    def __str__(self):
        return self.name + ': None (dummy gate, codebook size: ' + len(self.codebook[:, 0, 0]) + ')'


# This solver does a fixed point iteration. Simplest solver thinkable
# this dramatically reduced. Essentially only solve the propagate function
class nor_circuit_solver_banach:
    def __init__(self, circuit, values, bound=False):
        self.circuit = circuit
        if values is not None:
            self.values = np.copy(values)
        else:
            self.values = np.zeros(len(self.circuit))
        self.bound = bound

    def set_initial_value(self, initial_value):
        self.values = initial_value

    def bounding_mode(self, bound, heuristic=False):
        self.heuristic = heuristic
        self.bound = bound

    def solve(self, tol=10 ** (-2), max_iter=100):
        # Iterate the points using Banach's fixed point theorem
        # to estimate the error, compare old and new node values
        err = np.inf
        run = 0
        if self.bound:
            function = (lambda a: self.circuit.propagate_bound(a, heuristic=self.heuristic))
        else:
            function = self.circuit.propagate
        vs = self.values
        while err > tol:
            if (DEBUG_LEVEL > 1):
                self._debug_step(vs, run, err)
            new_vs = function(vs)
            # error and max_iter updates
            err = np.max(np.abs(new_vs - vs))
            vs = new_vs
            run += 1
            if run >= max_iter:
                break
        if (DEBUG_LEVEL > 0):
            self._debug_step(vs, run, err)
        self.values = vs
        return (vs, err, run)

    def _debug_step(self, vs, run, err):
        print('---------\n' + head('Outputs') + ' (run ' + hl(str(run)) + ', err < ' + hl(str(err)) + ', logic = ' + hl(
            str(self.circuit.bound_env[self.circuit.node_idx[list(self.circuit.structure.outputs)[0]]])) + '): ')
        for k, v in self.circuit.node_idx.items():
            if self.circuit.gates[self.circuit.node_idx[k]].type == 0:
                print('{k:<30}'.format(k=hl(k) + ' (env = ' + str(self.circuit.bound_env[v]) + ')') + ': ' + hl(
                    str(vs[v])))
            elif self.circuit.gates[self.circuit.node_idx[k]].type == -1:
                print('{k:<30}'.format(k=hl(k)) + ': dummy')
            elif self.circuit.gates[self.circuit.node_idx[k]].type == 1:
                print('{k:<30}'.format(k=hl(k) + ' (implicit or)') + ': ' + hl(str(vs[v])))


# This solver uses Newton's method. Reliable and fast (quadratic convergence)
class circuit_solver_newton:
    def __init__(self, inputs, tfs, gates):
        pass


# ____________________________________________________________________________
#   File interfacing classes
# ____________________________________________________________________________

class json_file:
    def __init__(self, path, read=True):
        self.path = path
        self.content = None
        if read:
            self.read()

    def read(self, update=False):
        if self.content is None or update == True:
            with open(self.path, 'r') as fyle:
                self.content = fyle.read().replace('\n', '')  # json.load(fyle)

    def write(self):
        with open(self.path, 'w') as fyle:
            self.content = fyle.write(self.content)  # json.dump(fyle)


# ____________________________________________________________________________
#   Classes and functions for more abstract objects/tasks
# ____________________________________________________________________________

class __directed_acyclic_graph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def is_valid(self):
        pass


def __list_intersect(a, b):
    return list(set(a) & set(b))


def _filter_dict(d, l, as_list=False, act=copy):
    if as_list:
        return [act(d[ll]) for ll in l if ll in d]
    return {ll: act(d[ll]) for ll in l if ll in d}


def dict_like(d, fill=None):
    if not isinstance(v, dict):
        return copy(fill)
    return dict({k: dict_like(v, fill) for k, v in d.items()})


def dict_from_list(l, fill=None):
    return dict({k: copy(fill) for k in l})


# JIT friendly version of integer power
def __power(bases, exponents):
    for n in range(len(bases)):
        b = bases[n]
        e = exponents[n] - 1
        for m in range(e):
            bases[n] *= b
