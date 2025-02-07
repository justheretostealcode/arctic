import cProfile
from collections import OrderedDict

import numpy as np
from scipy.interpolate import RegularGridInterpolator, CubicSpline
from deprecated import deprecated

from simulator.circuit_utils import CircuitStructure, CircuitAssignment
from simulator.gatelib import Promoter, UTR, CodingSequence, GateLibCollectionBased
from simulator.utils import JsonFile


@deprecated("This class is to be replaced by GeneticLogicCircuit")
class Circuit:
    @deprecated
    def __init__(self, structure: CircuitStructure):
        self.structure = structure
        structure.combinational_order()

        self.propagation_graph = None  # Currently propagation graph is equal to structure.adjacency["out"]
        self.inputs_graph = None
        self.generate_graph()

        self.assignment = None

        self.energy_rate = np.nan

        pass

    def generate_graph(self):

        structure = self.structure
        propagation_graph = OrderedDict()
        inputs_graph = OrderedDict()

        # {gate_id: iX for iX, gate_id in enumerate(self.structure.combinational_order())}

        for gate in structure.combinational_order():
            propagation_graph[gate] = [edge[1] for edge in structure.edges if edge[0] == gate]
            inputs_graph[gate] = [edge[0] for edge in structure.edges if edge[1] == gate]

        self.propagation_graph = propagation_graph
        self.inputs_graph = inputs_graph
        pass

    def set_assignment(self, assignment: CircuitAssignment):
        self.assignment = assignment
        self.energy_rate = np.nan
        pass

    def __call__(self, input_vals_dict, sim_settings):

        assignment = self.assignment
        propagation_graph = self.propagation_graph
        # inputs_graph = self.inputs_graph
        gate_input_vals = {id: [] for id in propagation_graph}
        gate_output_vals = {id: np.nan for id in propagation_graph}

        energy_rate = 0
        energy_rates = np.zeros(3)

        # Insert Input Values
        for input_id in input_vals_dict:
            gate_input_vals[input_id].append(input_vals_dict[input_id])

        # Propagate values through circuit
        for gate_id in propagation_graph:
            node_info = self.structure.node_infos[gate_id]
            device = assignment(node_info)
            # str(device)
            in_vals = gate_input_vals[gate_id]
            assert not (any(np.isnan(in_vals)) or any([val < 0 for val in in_vals]))

            out_val = device(*in_vals, sim_settings=sim_settings)
            gate_output_vals[gate_id] = out_val

            # Propagate device output to subsequent gates
            for subsidary in propagation_graph[gate_id]:
                gate_input_vals[subsidary].append(out_val)

            # if node_info.type == "LOGIC":
            energy_rate += device.energy_rate
            energy_rates += np.array(device.energy_rates)

            # print("")
            # print(gate_id)
            # print(gate_input_vals)
            # print(gate_output_vals)
            # pass

        self.energy_rate = energy_rate
        self.energy_rates = energy_rates

        return gate_output_vals


# The task of this class is to provide an abstraction layer on the layer of genetic logic gates
class GeneticLogicCircuit:
    def __init__(self, structure: CircuitStructure, settings: dict = None):
        self.structure = structure
        structure.combinational_order()

        self.settings = settings

        # self.propagation_graph = None
        # self.generate_graph()

        self.assignment = None
        self.input_mapping = None

        self.cell_state = None
        self.energy_rate = np.nan
        self.energy_rates = None
        self.energy_per_gate = None
        # Derives gene circuit's structure from genetic gate circuit's structure
        genes, gene_propagation_graph = self.setup_gene_circuit_structure()
        self.genes = genes
        self.gene_propagation_graph = gene_propagation_graph
        self.gene_circuit = GeneCircuit(genes=genes,
                                        gene_propagation_graph=gene_propagation_graph,
                                        settings=settings)


        pass

    def __call__(self, input_vals_dict, sim_settings):
        mode = sim_settings["mode"]
        n_samples = 1 if mode == "det" else sim_settings["n_samples"]
        sim_settings["n_samples_simulation"] = n_samples

        # Translate input_vals_dict consisting of the boolean input value to
        # input vals of the Gene Circuit, which are inducer concentrations
        gene_circuit_inputs = {}
        for input_id in input_vals_dict:
            input_mapping = self.input_mapping[input_id]
            inducer_name = input_mapping["inducer_name"]
            mapping = input_mapping["mapping"]
            # Here, the scalar input vals are mapped to arrays of the inducer concentrations
            gene_circuit_inputs[inducer_name] = np.ones(n_samples) * mapping[input_vals_dict[input_id]]

        # Perform the actual evaluation of the gene_circuit
        cell_state = self.gene_circuit(input_vals_dict=gene_circuit_inputs, sim_settings=sim_settings)

        # Cell State includes protein levels and energy consumption
        # However, the quantities of interest for the genetic logic circuit are promoter activities.
        # The subsequent extracts the information of interest.
        energy_per_gate = {}
        gate_output_vals = {}
        for gate_id in self.structure.nodes:
            genes = self.genes_by_associated_promoter[gate_id]
            # When the promoter of multiple genes are associated to a single genetic gate device,
            # the gate drives multiple subsequent gates. However, the "output" (given by the promoter activity)
            # of the genetic gate should be equal for all subsequent gates. In turn a single value is sufficient.

            # The output of the genetic gate is in RPU, the output of the gene is in gene expression units.
            # However, the ####_rpu entry refers to the RPU derived for the protein level outputted
            # As so, the prot_name_rpu of the cds accompanying the promoter on the gene is the required value
            cur_promoter_activity = np.zeros(n_samples)
            for gene in genes:
                # prot_name = gene.cds.name
                # cur_promoter_activity += cell_state[prot_name + "_rpu"]
                # cur_promoter_activity += gene.gene_state["average_promoter_activity"]
                # The current value isn't the actual promoter activity but the rescaled protein level
                cur_promoter_activity += gene.gene_state["protein_level"] / gene.cds.model.scaling_factor

            num_genes = len(genes)
            gate_output_vals[gate_id] = cur_promoter_activity / num_genes if num_genes > 1 else cur_promoter_activity
            # if len(genes) > 0:
            #     cur_promoter_activity = genes[0].gene_state["average_promoter_activity"]
            #     gate_output_vals[gate_id] = cur_promoter_activity
            # gate_info = self.structure.node_infos[gate_id]
            # if "OUTPUT" in gate_info.type:
            #     device = self.assignment(node_info=gate_info)
            #     gate_output_vals[gate_id] = cell_state[device.cds.name]

            energy_per_gate[gate_id] = {key: np.zeros(n_samples) for key in ["promoter", "rna", "protein"]}
            for gene in genes:
                energy_per_gate[gate_id]["promoter"] += gene.gene_state["energy"]["promoter"]

            genes = self.genes_by_associated_cds[gate_id]
            for gene in genes:
                energy_per_gate[gate_id]["rna"] += gene.gene_state["energy"]["rna"]
                energy_per_gate[gate_id]["protein"] += gene.gene_state["energy"]["protein"]

            gate_info = self.structure.node_infos[gate_id]
            if "OUTPUT" in gate_info.type:
                device = self.assignment(node_info=gate_info)
                gate_output_vals[gate_id] = cell_state[device.cds.name + "_rpu"]
        # ToDo translate from Gene Expression Model back to Genetic Logic Circuit
        # raise Exception("Output Node is currently Missing!!!")
        energy_consumption = self.gene_circuit.energy_consumption
        self.energy_rate = -cell_state["energy"]
        self.energy_rates = np.array([energy_consumption[key] for key in ["promoter", "rna", "protein"]])
        self.energy_per_gate = energy_per_gate
        self.cell_state = cell_state
        return gate_output_vals

    # def generate_graph(self):
    #     structure = self.structure
    #     propagation_graph = OrderedDict()
    #
    #     # {gate_id: iX for iX, gate_id in enumerate(self.structure.combinational_order())}
    #
    #     for gate in structure.combinational_order():
    #         propagation_graph[gate] = [edge[1] for edge in structure.edges if edge[0] == gate]
    #
    #     self.propagation_graph = propagation_graph
    #     pass

    def set_assignment(self, assignment: CircuitAssignment):
        self.assignment = assignment
        self.energy_rate = np.nan
        self.energy_rates = None
        # ToDo Check correctness
        gene_assignment = self.translate_assignment(assignment)
        self.gene_circuit.assign_genes(gene_assignment)
        pass

    # Maps a genetic gate circuit structure to a gene circuit structure
    def setup_gene_circuit_structure(self):
        if self.structure is None:
            return

        # Each genetic gate consists of a CDS promoter pair for a single transcription factor (TF)
        # The signal carrier is the promoter activity,
        # Normal Case:
        #       Each edge corresponds to a signal transmission realized by the output of a promoter.
        #       Because of this, src_id defines the promoter of the gene expressing the transcription
        #       factor associated to tar_id. Since no tandem promoters exist, the number of genes equals
        #       the number of promoters and so the number of edges.
        # Tandem Promoter Case:
        #       Again, the signal transmission is realized by the promoter activity. However, this time
        #       NOR Gates are realized by tandem promoters as inputs. In turn, two promoters are associated
        #       to a single gene.
        #       THIS CASE IS CURRENTLY NOT SUPPORTED
        #       (It could be energetically beneficial as only a single TF is expressed by two Promoters)
        # Implicit OR:
        #       The implicit OR only lacks a promoter giving rise to the gate's output value as the reporter
        #       protein produced is the output. In case

        genes = {}
        genes_by_associated_promoter = {gate_id: [] for gate_id in self.structure.nodes}
        genes_by_associated_cds = {gate_id: [] for gate_id in self.structure.nodes}
        for iD, (src_id, tar_id) in enumerate(self.structure.edges):
            # src_id defines promoter
            # tar_id defines cds
            gene = Gene(id=f"Gene {iD}",
                        promoter_associated_device=src_id,
                        cds_associated_device=tar_id,
                        settings=self.settings)
            genes[f"Gene {iD}"] = gene

            genes_by_associated_promoter[src_id].append(gene)
            genes_by_associated_cds[tar_id].append(gene)

        # Derive Propagation Graph
        gene_propagation_graph = {}
        for gene_id in genes:
            gene = genes[gene_id]
            cds_device = gene.cds_associated_device
            target_gene = genes_by_associated_promoter[cds_device]
            gene_propagation_graph[gene_id] = target_gene

        self.genes = genes
        self.gene_propagation_graph = gene_propagation_graph

        # Variables required for abstraction between GeneticLogicCircuit and GeneCircuit
        self.genes_by_associated_promoter = genes_by_associated_promoter
        self.genes_by_associated_cds = genes_by_associated_cds

        return genes, gene_propagation_graph

    def translate_assignment(self, assignment: CircuitAssignment):
        if self.genes is None or assignment is None:
            return None
        structure = self.structure
        assignment = self.assignment

        genes_by_associated_promoter = self.genes_by_associated_promoter
        genes_by_associated_cds = self.genes_by_associated_cds

        # ToDo Check correctness of output Assignment

        # Assignment includes mapping from gates to devices (genetic gates)
        # Gene Assignment shall include mapping from genes to gene parts
        # The gene inputs additionally require TF_inputs

        gene_assignment = {gene_id: {} for gene_id in self.genes}
        input_mapping = {input_id: None for input_id in structure.inputs}
        for gate_id in structure.nodes:

            gate_info = structure.node_infos[gate_id]
            device = assignment(gate_info)

            associated_promoters = genes_by_associated_promoter[gate_id]
            associated_cds = genes_by_associated_cds[gate_id]

            for gene in associated_promoters:
                gene_assignment[gene.id]["promoter"] = device.promoter
            for gene in associated_cds:
                gene_assignment[gene.id]["utr"] = device.utr
                gene_assignment[gene.id]["cds"] = device.cds
                gene_assignment[gene.id]["terminator"] = device.terminator

            # if gate_info.type == "INPUT":
            #     protein_input = device.cds
            #     # protein_input.input_id = gate_id    # Inform the Input to which value it corresponds.
            #     input_mapping[gate_id] = protein_input.signal_name
            #     gene_assignment[protein_input.name] = protein_input

            if gate_info.type == "INPUT":
                inducer_name = device.promoter.cognate_transcription_factors[0]
                input_mapping[gate_id] = {"inducer_name": inducer_name,
                                          "mapping": device.promoter.bool_to_inducer_lut}

        self.input_mapping = input_mapping
        return gene_assignment
        # if not self.is_valid():
        #     raise Exception("Assignment yielded invalid genetic circuit.")


class GeneCircuit:
    def __init__(self, genes: dict, gene_propagation_graph: dict, gene_assignment: dict = None, settings: dict = None):

        self._gene_assignment = gene_assignment
        self.genes = genes
        self.gene_propagation_graph = gene_propagation_graph
        self.settings = settings

        self.gene_evaluation_order = self.get_gene_evaluation_order()

        self.energy_consumption = None
        self.cell_state = None
        self.input_models = []
        pass

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, structure):
        self._structure = structure
        self.create_gene_circuit()

    @property
    def gene_assignment(self):
        return self._gene_assignment

    @gene_assignment.setter
    def gene_assignment(self, gene_assignment):
        if gene_assignment is None:
            return

        self._gene_assignment = gene_assignment
        self.input_models = []
        for elem_id in gene_assignment:
            if elem_id in self.genes:
                gene_id = elem_id
                gene = self.genes[gene_id]
                if gene_id not in gene_assignment:
                    raise Exception(f"Gene {gene_id} is not assigned any parts.")
                cur_assignment = gene_assignment[gene_id]
                # gene.promoter = cur_assignment["promoter"]
                # gene.utr = cur_assignment["utr"]
                # gene.cds = cur_assignment["cds"]
                # gene.terminator = cur_assignment["terminator"]
                gene.assign_parts(cur_assignment)

            else:  # The current element is an input
                self.input_models.append(gene_assignment[elem_id])

    def assign_genes(self, gene_assignment):
        self.gene_assignment = gene_assignment

    def is_valid(self):
        valid = True
        for gene_id in self.genes:
            valid = valid and self.genes[gene_id].is_valid()

        return valid

    def get_gene_evaluation_order(self):
        genes = self.genes
        gene_propagation_graph = self.gene_propagation_graph
        # Derive Gene Evaluation Order
        gene_evaluation_order = [None] * len(genes)
        unvisited_genes = set(genes.keys())
        visited_genes = set()
        iPos = len(genes) - 1
        while len(unvisited_genes) > 0:
            cur_unvisited_genes = set(unvisited_genes)
            for src_gene in cur_unvisited_genes:
                target_genes = gene_propagation_graph[src_gene]
                target_gene_ids = set(map(lambda gene: gene.id, target_genes))
                diff = target_gene_ids.difference(visited_genes)
                if len(diff) == 0:
                    gene_evaluation_order[iPos] = genes[src_gene]
                    iPos -= 1
                    visited_genes.add(src_gene)
                    unvisited_genes.remove(src_gene)

        return gene_evaluation_order

    def __call__(self, input_vals_dict: dict, sim_settings: dict, *args, **kwargs):
        # # Propagate values through the Gene Circuit
        # structure = self.structure
        # # connection_graph = structure.adjacency["out"]
        # propagation_graph = self.gene_propagation_graph

        n_samples = sim_settings["n_samples_simulation"]

        # All quantities relevant for the genetic circuit evaluation are stored in this dict
        # Proteins
        # RNA
        # Small Molecules
        cell_state = {"energy": np.zeros(n_samples)}
        cell_state.update(input_vals_dict)
        energy_consumption = OrderedDict({"promoter": np.zeros(n_samples),
                                          "rna": np.zeros(n_samples),
                                          "protein": np.zeros(n_samples),
                                          "overall": np.zeros(n_samples)})
        # for input_model in self.input_models:
        #     # Updates cell state by inserting the respective TF amount into cell_state
        #     input_model(cell_state, sim_settings)

        # gate_input_vals = {}
        # gate_output_vals = {}
        # transcription_factor_levels = {}
        for gene in self.gene_evaluation_order:
            # Update Gene's state with the cell state
            # The Gene updates the cell state itself
            output_dict = gene(cell_state, sim_settings=sim_settings)

            for key in energy_consumption:
                energy_consumption[key] += output_dict["energy"][key]

            # """
            # # output_dict format
            # output = {"protein_level": protein_level,
            #           "energy": {"promoter": energy_promoter,
            #                      "rna": energy_rna,
            #                      "protein": energy_protein}}
            # """

        self.cell_state = cell_state
        self.energy_consumption = energy_consumption
        return cell_state


# This entity is simulated with the gene expression model
class Gene:
    func_ids = ["average_promoter_activity",
                "rna_mean",
                "rna_var",
                "protein_mean",
                "protein_var",
                "protein_rna_covariance",
                "energy_p",
                "energy_tx",
                "energy_tl"]

    def __init__(self, id: str, promoter_associated_device: str = None, cds_associated_device: str = None,
                 promoter: Promoter = None, utr: UTR = None, cds: CodingSequence = None, terminator=None,
                 settings: dict = None):
        self.id = id
        self.promoter_associated_device = promoter_associated_device
        self.cds_associated_device = cds_associated_device
        self.promoter = promoter
        self.utr = utr
        self.cds = cds
        self.terminator = terminator
        # self.settings = settings

        # Approximation of relevant quantities via cubic splines
        # There are 3 energy quantities and 5 moment quantities of interest.
        self.order = None
        self.interpolators = None
        self.initialized_interpolators = False

        self.gene_state = None

    def __str__(self):
        return f"{self.id} (P {self.promoter_associated_device} -> CDS {self.cds_associated_device}): P={self.promoter} UTR={self.utr} CDS={self.cds} T={self.terminator}"

    def __call__(self, cell_state: dict, sim_settings: dict, *args, **kwargs):

        mode = "det"
        if "mode" in sim_settings:
            mode = sim_settings["mode"]

        n_samples = sim_settings["n_samples_simulation"]
        # interpolate = 0
        # if "interpolate" in sim_settings:
        #     interpolate = sim_settings["interpolate"]

        input_val_dict = {"c": np.sum(np.array([cell_state[molecule]
                                                for molecule in cell_state
                                                if molecule in self.promoter.cognate_transcription_factors]), axis=0)}
        # input_val_dict = {"c": sum([cell_state[molecule + "_rpu"]
        #                             for molecule in cell_state
        #                             if molecule in self.promoter.cognate_transcription_factors])}
        # ToDo Obtain relevant molecules from the cell_state
        # Transform the output of the model into information compatible with the cell state
        # Insert model output into the cell state

        # ToDo: Add support for LUT Promoter
        #  (Alternative -> Change LUT Promoter to normal Promoter and Realize Input by TF Lookup)
        #  This would be far more realistic

        scaling_factor = self.cds.model.scaling_factor
        # if interpolate:
        #     # In the approximate case, the actual response characteristic is approximated by interpolation
        #     #
        #     interpolation_output = self.perform_interpolation(input_val_dict, sim_settings=sim_settings)
        #     model_outputs = interpolation_output
        #
        # else:
        promoter_output = self.promoter(input_val_dict, sim_settings)
        cds_output = self.cds(promoter_output, sim_settings)
        model_outputs = cds_output

        # mean_M = model_outputs["rna_mean"]
        # var_M = model_outputs["rna_var"]
        mean_P = model_outputs["protein_mean"]
        var_P = model_outputs["protein_var"]
        # covar_protein_rna = model_outputs["protein_rna_covariance"]

        energy_promoter = model_outputs["energy_p"]
        energy_rna = model_outputs["energy_tx"]
        energy_protein = model_outputs["energy_tl"]

        if mode == "samp":
            # The function value is sampled from the distribution
            common_val = var_P / (mean_P ** 2) + 1
            common_val[common_val < 0] = 0
            sigma = np.sqrt(np.log(common_val))
            mu = np.log(mean_P ** 2 / np.sqrt(var_P + mean_P ** 2))

            sample = np.random.lognormal(mu, sigma)

            protein_level = sample
        elif mode == "det-var":
            protein_level = mean_P
        elif mode == "det":
            protein_level = mean_P
        else:
            raise Exception(f"Mode {mode} is not supported.")

        energy_gene = energy_promoter + energy_rna + energy_protein
        gene_state = {"protein_level": protein_level,
                      "energy": {"promoter": energy_promoter,
                                 "rna": energy_rna,
                                 "protein": energy_protein,
                                 "overall": energy_gene}}
        gene_state.update(model_outputs)

        # Update cell's state
        prot_name = self.cds.name
        if prot_name not in cell_state:
            cell_state[prot_name] = np.zeros(n_samples)
            cell_state[prot_name + "_rpu"] = np.zeros(n_samples)
        cell_state[prot_name] += protein_level
        cell_state[prot_name + "_rpu"] += protein_level * 1 / scaling_factor

        # Update cell's energy level
        cell_state["energy"] -= energy_gene

        # Update own_state and return it
        self.gene_state = gene_state

        return gene_state

    def assign_parts(self, parts_assignment):
        self.promoter = parts_assignment["promoter"]
        self.utr = parts_assignment["utr"]
        self.cds = parts_assignment["cds"]
        self.terminator = parts_assignment["terminator"]
        self.initialized_interpolators = False

        # if self.settings is not None and self.settings["interpolate"]:
        # self.interpolators = self.init_interpolation()

    def _in_vals_dict_to_array(self, in_vals_dict):
        if self.order is None:
            self.order = list(in_vals_dict.keys())
        return [in_vals_dict[elem] for elem in self.order]

    def is_valid(self):
        valid = True
        valid = valid and self.promoter is not None
        # valid = valid and self.utr is not None
        valid = valid and self.cds is not None
        # valid = valid and self.promoter is not None
        return valid

    # def init_interpolation(self) -> dict:
    #     # raise Exception("Not ")
    #     # return {}
    #     input_vals = np.logspace(-4, 2, 100)  # ToDo Define adequate interval
    #     input_vals_dicts = [{"c": val} for val in input_vals]
    #
    #     sim_settings = {"mode": "samp"}
    #
    #     interpolation_data = {func_id: np.empty(shape=len(input_vals_dicts)) for func_id in Gene.func_ids}
    #     X = [None] * len(input_vals_dicts)
    #     for iX, input_vals_dict in enumerate(input_vals_dicts):
    #         promoter_output = self.promoter(input_vals_dict, sim_settings)
    #         cds_output = self.cds(promoter_output, sim_settings)
    #
    #         for func_id in Gene.func_ids:
    #             interpolation_data[func_id][iX] = cds_output[func_id]
    #
    #         X[iX] = self._in_vals_dict_to_array(input_vals_dict)
    #
    #     X = np.array(X)
    #     X = X.transpose()
    #     interpolators = {}
    #     for func_id in Gene.func_ids:
    #         Y = interpolation_data[func_id]
    #         # interpolator = RegularGridInterpolator(X, Y, method="linear", bounds_error=True, fill_value=None)
    #         interpolator = CubicSpline(X.squeeze(), Y)
    #         # interpolation = interpolator(xi=[[0.5], [1]])
    #         interpolators[func_id] = interpolator
    #
    #     self.initialized_interpolators = True
    #
    #     return interpolators
    #
    # def perform_interpolation(self, input_vals_dict, sim_settings):
    #     if not self.initialized_interpolators:
    #         return None
    #
    #     in_vals = self._in_vals_dict_to_array(input_vals_dict)
    #
    #     interpolation_output = {}
    #     for func_id in Gene.func_ids:
    #         try:
    #             # interpolation_output[func_id] = self.interpolators[func_id]([in_vals])[0]
    #             interpolation_output[func_id] = self.interpolators[func_id](in_vals)
    #         except:
    #             interpolation_output[func_id] = 1111
    #
    #     return interpolation_output


if __name__ == '__main__':
    json_structure = JsonFile(path="ARCTICsim/simulator_nonequilibrium/data/structures/structure_01110101.json")
    json_structure = JsonFile(
        path="ARCTICsim/simulator_nonequilibrium/data/structures/11111000_structure_0.json")  # Features an implicit OR
    # json_gatelib = JsonFile(path="ARCTICsim/simulator_nonequilibrium/data/gate_libs/gate_lib_yeast.json")
    # json_gatelib = JsonFile(path="ARCTICsim/simulator_nonequilibrium/data/gate_libs/gate_lib_draft.json")
    json_gatelib = JsonFile(
        path="ARCTICsim/simulator_nonequilibrium/data/gate_libs/gate_lib_yeast_generated_mean_fit.json")
    # gate_lib = GateLib(json_gatelib)
    gate_lib = GateLibCollectionBased(json_file=json_gatelib)

    structure = CircuitStructure(json_structure)
    genetic_logic_circuit = GeneticLogicCircuit(structure=structure, settings={"interpolate": 0})
    # gene_circuit = GeneCircuit(structure=structure, assignment=assignment)

    # assignment_json = '{"a":"input_3","b":"input_1","c":"input_2","NOT_0":"H1_HlyIIR","NOT_2":"S2_SrpR","NOT_4":"B3_BM3R1","NOR2_1":"L1_LitR","NOR2_3":"P2_PhlF","O":"output_1"}'
    assignment = None
    # json_assignment = JsonFile(path="ARCTICsim/simulator_nonequilibrium/data/assignments/assignment_01110101.json")
    # json_assignment = JsonFile(path="ARCTICsim/simulator_nonequilibrium/data/assignments/dummy_assignment_11111000_structure_0.json")
    json_assignment = JsonFile(
        path="ARCTICsim/simulator_nonequilibrium/data/assignments/test_assignment_11111000_structure_0.json")
    assignment = CircuitAssignment(json_assignment, gate_lib=gate_lib)
    assignment(genetic_logic_circuit.structure.node_infos["a"])

    genetic_logic_circuit.set_assignment(assignment)
    # circuit = Circuit(structure)
    # circuit.set_assignment(assignment)
    # print(circuit.propagation_graph)
    genetic_logic_circuit.genes
    # input_vals = {"input_1": 10 ** 9, "input_2": 10 ** 3, "input_3": 10 ** 8}
    test_gene = genetic_logic_circuit.genes_by_associated_promoter["NOT_1"][0]
    cell_state = {"energy": 0, "CI": np.ones(2) * 1.1 * 10 ** 5}
    test_gene(cell_state=cell_state,
              sim_settings={"mode": "samp", "n_samples": 2, "n_samples_simulation": 2})
    input_vals = {"a": 0, "b": 1, "c": 0}
    # output_vals = circuit(input_vals_dict=input_vals, sim_settings={})
    output_vals = genetic_logic_circuit(input_vals_dict=input_vals, sim_settings={"mode": "det", "interpolate": False})
    print(output_vals)

    print(genetic_logic_circuit.energy_rate)
    print(genetic_logic_circuit.energy_rates)

    print(structure.to_dot())

    num_particles = 1000 * 4
    N = 8 * num_particles  # Simulates 1000 Particles for a three input circuit
    input_vals_list = [{"a": 0, "b": 0, "c": 0},
                       {"a": 0, "b": 0, "c": 1},
                       {"a": 0, "b": 1, "c": 0},
                       {"a": 0, "b": 1, "c": 1},
                       {"a": 1, "b": 0, "c": 0},
                       {"a": 1, "b": 0, "c": 1},
                       {"a": 1, "b": 1, "c": 0},
                       {"a": 1, "b": 1, "c": 1}]

    profiler = cProfile.Profile()
    profiler.enable()
    # out_vals = np.empty(N)
    for iX in range(N):
        input_vals = input_vals_list[iX // num_particles]
        output_vals = genetic_logic_circuit(input_vals_dict=input_vals, sim_settings={"mode": "samp", "interpolate": 0})
        # out_vals[iX] = output_vals["O"]

    profiler.disable()
    profiler.dump_stats("profile_results_exact.prof")
    profiler.print_stats()

    print("")
    profiler = cProfile.Profile()
    profiler.enable()

    # out_vals = np.empty(N)
    for iX in range(N):
        output_vals = genetic_logic_circuit(input_vals_dict=input_vals, sim_settings={"mode": "samp", "interpolate": 0})
        # out_vals[iX] = output_vals["O"]

    profiler.disable()
    profiler.dump_stats("profile_results_interpolation.prof")
    profiler.print_stats()
