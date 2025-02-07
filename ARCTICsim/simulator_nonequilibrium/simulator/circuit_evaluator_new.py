"""
Author: Erik Kubaczka
"""
import numpy as np
import time

from matplotlib import pyplot as plt

from simulator.circuit import GeneticLogicCircuit
from simulator.scores import FunctionalScore, EnergyScore
from simulator.circuit_utils import CircuitAssignment, CircuitStructure


# ToDos
# 1. Implement Interface    ✓
#   - Load Stucture             ✓
#   - Load Gate Lib             ✓
#   - Interpret Assignment      ✓
#
# 2. Implement Function
#   - Construct assigned circuit
#   - Implement simulation (based on samples from distributions)
#   - Propagate values through circuit
#
# 3. Perform scoring
#   - Functional Scoring based on Wasserstein
#   - Energy Scoring (either Average or Maximum)
#
# 4. Return scores              ✓
#   - Pass scores in JSON Format to optimizer           ✓

class CircuitEvaluator:

    def __init__(self, gate_lib, settings, structure: CircuitStructure):
        self.gate_lib = gate_lib
        self.settings = settings
        self.set_structure(structure)

        self.update_settings(settings)

        # print("Use Python Implementation")
        pass

    def set_structure(self, structure: CircuitStructure):
        if structure is None:
            return

        self.structure = structure
        self.circuit = GeneticLogicCircuit(structure, self.settings)

        # Generate Truthtable

    def update_settings(self, settings: dict):
        self.settings = settings

        self.functional_score = FunctionalScore(settings)
        self.energy_score = EnergyScore(settings)

        self.DEBUG_LEVEL = settings["verbosity"]

    def score_assignment(self, assignment: CircuitAssignment, sim_settings: dict):
        if "verbosity" in sim_settings:
            self.DEBUG_LEVEL = sim_settings["verbosity"]

        if self.DEBUG_LEVEL >= 1:
            pass

        circuit = self.circuit

        circuit.set_assignment(assignment)

        mode = sim_settings["mode"]
        n_samples = sim_settings["n_samples"] if mode == "samp" else 1

        structure = circuit.structure
        truthtable = structure.truthtable

        node_ids = list(structure.nodes)
        num_gates = len(node_ids)
        bins = np.logspace(-5, 2, max(int(n_samples / 4), 100))
        nrows = int(np.floor(np.sqrt(num_gates)))
        ncols = int(np.ceil(num_gates / nrows))

        n_input_assignments = len(truthtable.input_output_truthtable())
        gate_output_vals = [None] * n_input_assignments
        circuit_output_vals_dict = {out_id: [[] for _ in range(2)] for out_id in structure.outputs}
        circuit_output_vals = [None] * n_input_assignments
        circuit_energy_rates = [None] * n_input_assignments
        detailed_circuit_energy_rates = [None] * n_input_assignments
        energy_per_gate = [None] * n_input_assignments

        input_ids = list(structure.inputs)
        output_ids = list(structure.outputs)
        input_ids.sort()

        # node_order = structure.combinational_order()
        start = time.time()
        # for iIndex, truthtable_vals in enumerate(truthtable.input_output_truthtable()):

        for iIndex, (input_vals, output_val) in enumerate(truthtable.input_output_truthtable()):

            input_vals_dict = dict(zip(input_ids, input_vals))
            # gate_output_vals = {id: np.empty(shape=(n_samples)) for id in structure.nodes}
            # energy_rates = np.empty(shape=(n_samples))
            # detailed_energy_rates = np.empty(shape=(n_samples, 3))
            # for iN in range(n_samples):
            #
            #     cur_gate_output_vals = circuit(input_vals_dict=input_vals_dict, sim_settings=sim_settings)
            #     cur_energy_rate = circuit.energy_rate
            #     cur_detailed_energy_rates = circuit.energy_rates
            #
            #     for id in cur_gate_output_vals:
            #         gate_output_vals[id][iN] = cur_gate_output_vals[id]
            #
            #     energy_rates[iN] = cur_energy_rate
            #     detailed_energy_rates[iN] = cur_detailed_energy_rates

            cur_gate_output_vals = circuit(input_vals_dict=input_vals_dict, sim_settings=sim_settings)
            cur_energy_rate = circuit.energy_rate
            cur_detailed_energy_rates = circuit.energy_rates
            cur_energy_per_gate = circuit.energy_per_gate

            # gate_output_vals = cur_gate_output_vals
            gate_output_vals[iIndex] = cur_gate_output_vals
            energy_rates = cur_energy_rate
            detailed_energy_rates = cur_detailed_energy_rates

            cur_out_vals = []
            for out_id in output_ids:
                circuit_output_vals_dict[out_id][output_val].append(cur_gate_output_vals[out_id])
                cur_out_vals.append(cur_gate_output_vals[out_id])

            circuit_output_vals[iIndex] = cur_out_vals
            circuit_energy_rates[iIndex] = energy_rates
            detailed_circuit_energy_rates[iIndex] = detailed_energy_rates
            energy_per_gate[iIndex] = cur_energy_per_gate

            if self.DEBUG_LEVEL > 0:
                print(f"\nInput Combination {input_vals_dict} -> Logic Val {output_val}")
                for node_id in self.structure.node_infos:
                    node_info = self.structure.node_infos[node_id]
                    print(f"{node_id} ({node_info.type}): {cur_gate_output_vals[node_id]}")
                pass

                if self.DEBUG_LEVEL >= 3:
                    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=False, squeeze=False)
                    for iR in range(nrows):
                        for iC in range(ncols):
                            cur_index = iR * ncols + iC
                            if cur_index < len(node_ids):
                                ax = axes[iR, iC]
                                node_id = node_ids[cur_index]
                                cur_vals = cur_gate_output_vals[node_id]
                                bool_val = structure.gate_truthtables[node_id][iIndex]
                                ax.hist(cur_vals, bins=bins, density=False, color="red" if bool_val == 0 else "blue")
                                ax.set_title(f"{node_id} ({bool_val})")
                                ax.set_xscale("log")

                    plt.suptitle(f"{input_vals} -> {output_val}")
                    plt.tight_layout()
                    plt.show()

            # print(gate_output_vals)
            pass
        end = time.time()
        duration = end - start
        # print(f"Duration: {end - start}")

        circuit_output_vals = np.array(circuit_output_vals)
        circuit_energy_rates = np.array(circuit_energy_rates)
        detailed_circuit_energy_rates = np.array(detailed_circuit_energy_rates)

        functional_scores = {}
        critical_indexes = {}
        for iO, out_id in enumerate(circuit_output_vals_dict):
            cur_entry = circuit_output_vals_dict[out_id]
            functional_score = np.infty
            critical_index_off = -1
            critical_index_on = -1
            for iOn, dataON in enumerate(cur_entry[1]):
                for iOff, dataOFF in enumerate(cur_entry[0]):
                    cur_score = self.functional_score(dataON=dataON, dataOFF=dataOFF)
                    if cur_score < functional_score:
                        functional_score = cur_score
                        critical_index_on = iOn
                        critical_index_off = iOff

            functional_scores[out_id] = functional_score
            critical_indexes[out_id] = {0: critical_index_off, 1: critical_index_on}

            if self.DEBUG_LEVEL >= 2:
                print("Extreme Vals:", np.median(cur_entry[0][critical_index_off]),
                      np.median(cur_entry[1][critical_index_on]))
                fig, axes = plt.subplots(nrows=len(truthtable.truthtable), sharex=True, sharey=False)
                for iE, elem in enumerate(circuit_output_vals):
                    ax = axes[iE]
                    ax.hist(elem[iO], bins=bins, density=False)
                    ax.set_title(truthtable.input_output_truthtable()[iE][-1])
                    ax.set_xscale("log")

                plt.suptitle(f"{out_id} {functional_score}")
                plt.tight_layout()
                plt.show()
        energy_score = self.energy_score(circuit_energy_rates)

        detailed_energy_score = {key: self.energy_score(detailed_circuit_energy_rates[:, iX, :]) for iX, key in
                                 enumerate(["e_p", "e_tx", "e_tl"])}

        scores = {"functional_score": functional_scores,
                  "energy_score": energy_score,
                  "detailed_energy_score": detailed_energy_score}

        if False:
            import pandas as pd
            import pathlib
            assignment_name = pathlib.Path(sim_settings["assignment"]).stem
            df = pd.DataFrame(energy_per_gate)
            df2 = df.map(lambda elem: sum(elem.values()).item())
            df2.to_excel(f"data/_output/energy_per_gate_{assignment_name}.xlsx", index=False)

            df = pd.DataFrame(gate_output_vals)
            df2 = df.map(lambda elem: sum(elem).item())
            df2.to_excel(f"data/_output/gate_output_vals_{assignment_name}.xlsx", index=False)


        return scores
