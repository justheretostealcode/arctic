import cProfile

import numpy as np

from models.moment_model_new import RNAMomentModel, ProteinMomentModel, CombinedMomentModel
from models.promoter_model_new import PromoterModel

from simulator.utils import JsonFile


# Class to represent the gatelib
# @deprecated("Replaced by GateLibCollectionBased")
# class GateLib:
#     def __init__(self, json_file: JsonFile):
#         #        raise Exception("Change to GateLibCollectionBased")
#         self.json = json_file
#
#         if json_file is None:
#             raise Exception("No Gate Library information provided!")
#
#         self.gates = []
#         for gate_entry in json_file.data:
#             gates = GateLib._populate_gate_entry(gate_entry)
#             self.gates += gates
#
#         self.gates_by_type_and_name = {}
#         for gate in self.gates:
#             gate_type = gate.type
#             ident = gate.identifier
#             if gate_type not in self.gates_by_type_and_name:
#                 self.gates_by_type_and_name[gate_type] = {}
#
#             if ident in self.gates_by_type_and_name[gate_type]:
#                 raise Exception(f"{ident} is already in the lookup dict")
#
#             self.gates_by_type_and_name[gate_type][ident] = gate
#         pass
#
#     @staticmethod
#     def _populate_gate_entry(gate_entry):
#         def _get_gate_implementation(gate_type):
#             if gate_type == "NOT":
#                 return NOTGate
#             elif gate_type == "NOR2":
#                 return NORGate
#             elif gate_type == "INPUT":
#                 return LutInput
#             elif gate_type == "OUTPUT_OR2":
#                 return OutputOR
#             elif gate_type == "OUTPUT_BUFFER":
#                 return OutputBuffer
#             else:
#                 raise Exception(f"Type \"{gate_type}\" not supported")
#             pass
#
#         gates = []
#
#         for gate_type in gate_entry["primitiveIdentifier"]:
#             gate_implementation = _get_gate_implementation(gate_type)
#
#             gate = gate_implementation(gate_entry)
#             gates.append(gate)
#
#         return gates


class GateLibCollectionBased:

    def __init__(self, json_file: JsonFile):
        self.json = json_file
        COLLECTIONS = {"promoters": Promoter,
                       "sensor_promoters": SensorPromoter,
                       "coding_sequences": CodingSequence,
                       "proteins": Protein,
                       "tf_inputs": TranscriptionfactorInputs,
                       "sequences": Sequence,
                       "utrs": UTR,
                       "terminators": Terminator,
                       "devices": Device}

        if json_file is None:
            raise Exception("No Gate Library information provided!")

        self.on_completion_callback = []

        """
        Gate Lib Format
        The new gate lib design is collection based
        
        The collections included are:
        - Promoter (and associated TF)
        - Transcription Factors
        - Sequences (Promoter and Transcription Factor)
        - Devices (Pairs of TF and Promoter with associated Logic Type Realized, Includes OutputBuffer and OutputOR2, LUT Inputs)
        
        
        Comment:
        As the inputs represent promoter activity, the LUT Inputs fall into the same category as the promoters.
        """
        self.objects = []
        self.objects_by_id = {}
        self.objects_by_collection = {key: [] for key in COLLECTIONS}
        self.objects_by_class = {COLLECTIONS[key]: [] for key in COLLECTIONS}
        for entry in self.json.data:
            print(entry)
            collection_type = entry["collection"]
            class_var = COLLECTIONS[collection_type]
            object = class_var(entry, gate_lib=self)
            self.objects.append(object)

            if object.id in self.objects_by_id:
                raise Exception(f"The id {object.id} is already included in:\n{self.objects_by_id}")
            self.objects_by_id[object.id] = object
            self.objects_by_collection[collection_type].append(object)
            self.objects_by_class[class_var].append(object)

        # self.gates = []
        # for gate_entry in json_file.data:
        #     gates = GateLib._populate_gate_entry(gate_entry)
        #     self.gates += gates
        #
        # self.gates_by_type_and_name = {}
        # for gate in self.gates:
        #     gate_type = gate.type
        #     ident = gate.identifier
        #     if gate_type not in self.gates_by_type_and_name:
        #         self.gates_by_type_and_name[gate_type] = {}
        #
        #     if ident in self.gates_by_type_and_name[gate_type]:
        #         raise Exception(f"{ident} is already in the lookup dict")
        #
        #     self.gates_by_type_and_name[gate_type][ident] = gate
        pass

        self.completion()

    def completion(self):
        for callback in self.on_completion_callback:
            callback(self)

    def add_on_completion_callback(self, callback):
        self.on_completion_callback.append(callback)

    def remove_on_completion_callback(self, callback):
        self.on_completion_callback.remove(callback)

    def __call__(self, id, object_class=None, *args, **kwargs):
        object = None
        if id in self.objects_by_id:
            object = self.objects_by_id[id]
        return object


class GateLibEntry:
    def __init__(self, collection_identifier: str = None,
                 gate_lib_entry_dict: dict = None,
                 gate_lib: GateLibCollectionBased = None):
        if collection_identifier is None:
            raise Exception("Collection Identifier is not set!")

        if gate_lib_entry_dict is None:
            raise Exception("No gate_lib_entry_dict provided.")

        self.collection_identifier = collection_identifier
        self.gate_lib_entry_dict = gate_lib_entry_dict
        self.id = gate_lib_entry_dict["identifier"]
        self.name = gate_lib_entry_dict["name"]
        self.collection = gate_lib_entry_dict["collection"]
        # self.group = gate_lib_entry_dict["group"] if "group" in gate_lib_entry_dict else None

        self.gate_lib = gate_lib

        if self.collection != self.collection_identifier:
            raise Exception(
                f"Mismatch in collection (Trying to instantiate {self.collection_identifier} with an entry of collection {self.collection})")

    def __str__(self):
        #return f"{self.collection_identifier} {self.id}"
        return f"{self.id}"


class GateLibModelEntry(GateLibEntry):
    def __init__(self, collection_identifier: str = None,
                 gate_lib_entry_dict: dict = None,
                 gate_lib: GateLibCollectionBased = None):
        super().__init__(collection_identifier=collection_identifier,
                         gate_lib_entry_dict=gate_lib_entry_dict,
                         gate_lib=gate_lib)

        self.model_info = gate_lib_entry_dict["model_info"]
        self.model = self._populate_model()

    def _populate_model(self):
        """
        Is implemented by the respective subclass to realize the model creation.
        :return: The model populated with the parameters provided in self.model_info
        """
        raise Exception("Not Implemented Yet")

    def __call__(self, in_vals: dict, sim_settings: dict, *args, **kwargs):
        """
        Is implemented by the respective subclass or the subclass provides a model matching the interface.
        :param args:
        :param kwargs:
        :return:
        """
        output = self.model(in_vals, sim_settings=sim_settings)
        return output


class Promoter(GateLibModelEntry):

    def __init__(self,
                 gate_lib_entry_dict: dict,
                 gate_lib: GateLibCollectionBased = None,
                 collection_identifier: str = None):
        self.collection_identifier = collection_identifier
        if collection_identifier is None:
            self.collection_identifier = "promoters"
        super().__init__(collection_identifier=self.collection_identifier,
                         gate_lib_entry_dict=gate_lib_entry_dict,
                         gate_lib=gate_lib)

        self.cognate_transcription_factors = gate_lib_entry_dict["cognate_transcription_factors"]
        self.sequence_ids = gate_lib_entry_dict["sequence_ids"]
        self.technology_mapping = gate_lib_entry_dict["technology_mapping"]
        # A single construct can be asssociated to multiple sequences (for example codon alternatives)

    def _populate_model(self):
        model_info = self.model_info
        num_states = model_info["N_PROMOTER_STATES"]
        num_trainable_parameters = model_info["N_PARAMETERS"]
        promoter_activity = model_info["PROMOTER_ACTIVITY"]
        infinitesimal_generator_function = model_info["INFINITESIMAL_GENERATOR_FUNCTION"]
        input_scaling_factor = model_info["INPUT_SCALING_FACTOR"]
        model = PromoterModel(num_states,
                              num_trainable_parameters,
                              input_scaling_factor=input_scaling_factor,
                              infinitesimal_generator_function=infinitesimal_generator_function,
                              per_state_promoter_activity=promoter_activity)

        return model


class SensorPromoter(Promoter):
    """
    This class represents the mapping from inducer to promoter activity.
    To this end, it employs static collapsed promoter models for each input to output relation.
    """

    def __init__(self, gate_lib_entry_dict: dict,
                 gate_lib: GateLibCollectionBased = None):
        self.collection_identifier = "sensor_promoters"
        self.bool_to_inducer_lut = None
        super().__init__(collection_identifier=self.collection_identifier,
                         gate_lib_entry_dict=gate_lib_entry_dict,
                         gate_lib=gate_lib)

        # This class misuses the field self.cognate_transcription_factors to represent the inducer it is sensitive to

    def _populate_model(self):
        model_info = self.model_info

        self.bool_to_inducer_lut = {0: model_info["low"],
                                    1: model_info["high"]}

        # Model Info provides a table providing the mapping of inducer concentration to RPU

        LUT = model_info["LUT"]
        val_LUT = {float(val): LUT[val] for val in LUT}
        values = np.array(list(val_LUT.items()))

        # model_LUT = {}
        # for input_val in LUT:
        #     num_states = 1
        #     num_trainable_parameters = 0
        #     promoter_activity = [LUT[input_val]]
        #     infinitesimal_generator_function = {"matrices": {"1": [[0]]}}
        #     input_scaling_factor = 1
        #     promoter_model = PromoterModel(num_states,
        #                                    num_trainable_parameters,
        #                                    input_scaling_factor=input_scaling_factor,
        #                                    infinitesimal_generator_function=infinitesimal_generator_function,
        #                                    per_state_promoter_activity=promoter_activity)
        #     model_LUT[float(input_val)] = promoter_model

        # look_up_func = lambda val: model_LUT[val]
        # look_up_func_vectorized = np.vectorize(look_up_func)

        # Setup linear interpolation
        diffs = np.diff(values, axis=0)
        steepnes = diffs[:, 1] / diffs[:, 0]
        offset = values[:-1, 1] - steepnes * values[:-1, 0]

        # def model_old(in_val_dict: dict, sim_settings: dict, *args, **kwargs):
        #     # Terribly slow despite the use of vectorize
        #     # Requires up to 0.6 s for 1000 particle simulation which is around 56 % of the total time
        #     def look_up_func(val):
        #         promoter_model = model_LUT[val]
        #         n_samples = sim_settings["n_samples_simulation"]
        #         sim_settings["n_samples_simulation"] = 1
        #         result = promoter_model({"c": np.zeros(1)}, sim_settings=sim_settings, *args, **kwargs)
        #         sim_settings["n_samples_simulation"] = n_samples
        #         return result
        #
        #     cognate_inducer_concentration = np.array(in_val_dict["c"])
        #     look_up_func_vectorized = np.vectorize(look_up_func)
        #     results = look_up_func_vectorized(cognate_inducer_concentration)
        #     result = {}
        #
        #     for key in results[0]:
        #         if key == "promoter_activity_per_state":
        #             continue
        #         accumulation = [elem[key] for elem in results]
        #         result[key] = np.concatenate(accumulation, axis=0)
        #
        #     result["promoter_activity_per_state"] = np.array([results[0]["promoter_activity_per_state"]])
        #     return result



        def get_promoter_info(val):

            promoter_info = {"distribution": np.array([[1]]),
                             "propensity_matrix": np.array([[[0]]]),
                             "entropy_production_rate":  np.array([0]),
                             "average_promoter_activity_per_state": np.array([[val]]),
                             "average_promoter_activity": np.array([val]),
                             "energy_dissipation_rate":  np.array([0]),
                             "energy_p":  np.array([0]),
                             "promoter_activity_per_state": np.array([val])}
            return promoter_info

        def model(in_val_dict: dict, sim_settings: dict, *args, **kwargs):

            quick_mode = "quick" in sim_settings and sim_settings["quick"]

            # Perform Interpolation
            cognate_inducer_concentration = np.array(in_val_dict["c"])
            cognate_inducer_concentration = np.expand_dims(cognate_inducer_concentration, -1)
            if quick_mode:
                interpolations = steepnes * cognate_inducer_concentration[:1] + offset
                # indicator_function_1 = cognate_inducer_concentration[:1] >= values[:, 0]
                # indicator_function_1 = np.expand_dims(indicator_function_1, -1)
                indicator_function_2 = cognate_inducer_concentration[:1] <= values[:, 0]
            else:
                interpolations = steepnes * cognate_inducer_concentration + offset
                # indicator_function_1 = cognate_inducer_concentration >= values[:, 0]
                indicator_function_2 = cognate_inducer_concentration <= values[:, 0]

            indicator_function = np.argmax(indicator_function_2, axis=1)
            interpolation_result = interpolations[np.arange(interpolations.shape[0]), indicator_function]

            if quick_mode:
                promoter_info = get_promoter_info(interpolation_result[0])
                results = [promoter_info] * cognate_inducer_concentration.shape[0]
            else:
                results = [get_promoter_info(val) for val in interpolation_result]

            result = {}

            for key in results[0]:
                if key == "promoter_activity_per_state":
                    continue
                accumulation = [elem[key] for elem in results]
                result[key] = np.concatenate(accumulation, axis=0)

            result["promoter_activity_per_state"] = np.array([results[0]["promoter_activity_per_state"]])
            return result

        return model


#
# class SmallMolecule(GateLibModelEntry):
#
#     def __init__(self, gate_lib_entry_dict: dict):
#         self.collection_identifier = "small_molecules"
#         super().__init__(collection_identifier=self.collection_identifier, gate_lib_entry_dict=gate_lib_entry_dict)
#
#         self.molecule_type = gate_lib_entry_dict["cognate_molecules"]
#         self.cognate_molecules = gate_lib_entry_dict["cognate_molecules"]
#
#     def _populate_model(self):
#         model_info = self.model_info
#         raise Exception("Model is currently not populated")


# class LutPromoter(GateLibModelEntry):
#     def __init__(self, gate_lib_entry_dict: dict):
#         self.collection_identifier = "lut_promoters"
#         super().__init__(collection_identifier=self.collection_identifier, gate_lib_entry_dict=gate_lib_entry_dict)
#
#         self.cognate_transcription_factors = gate_lib_entry_dict["cognate_transcription_factors"]
#         # The corresponding sequences are currently not included in the library.
#         self.sequence_idds = []
#
#     def _populate_model(self):
#         model_info = self.model_info
#         LUT = {float(key): model_info["LUT"][key] for key in model_info["LUT"]}
#
#         # ToDo Check whether this works as intended
#         # in_vals["c"] is the
#         def model(in_vals, sim_settings=None): LUT[in_vals["c"]]
#
#         return model


class CodingSequence(GateLibModelEntry):
    """
    Superclass of expresseble sequences.
    """

    def __init__(self, gate_lib_entry_dict: dict,
                 collection_identifier: str = None,
                 gate_lib: GateLibCollectionBased = None):
        self.collection_identifier = collection_identifier
        if collection_identifier is None:
            self.collection_identifier = "coding_sequences"
        super().__init__(collection_identifier=self.collection_identifier,
                         gate_lib_entry_dict=gate_lib_entry_dict,
                         gate_lib=gate_lib)

        self.sequence_ids = gate_lib_entry_dict["sequence_ids"]
        # A single construct can be asssociated to multiple sequences (for example codon alternatives)


class Protein(CodingSequence):
    def __init__(self, gate_lib_entry_dict: dict,
                 collection_identifier: str = None,
                 gate_lib: GateLibCollectionBased = None):
        self.collection_identifier = collection_identifier
        if collection_identifier is None:
            self.collection_identifier = "proteins"

        super().__init__(collection_identifier=self.collection_identifier,
                         gate_lib_entry_dict=gate_lib_entry_dict,
                         gate_lib=gate_lib)

    def _populate_model(self):
        model_info = self.model_info
        # Includes molecule dependent transcription, RNA degradation and the respective energy requirements and lengths.

        transcription_rate = model_info["rna"]["transcription_rate"]
        rna_degradation_rate = model_info["rna"]["degradation_rate"]
        energy_per_nucleotide = model_info["rna"]["e"]
        energy_per_rna = model_info["rna"]["e_const"]
        rna_length = model_info["rna"]["length"]

        translation_rate = model_info["protein"]["translation_rate"]
        protein_degradation_rate = model_info["protein"]["degradation_rate"]
        energy_per_amino_acid = model_info["protein"]["e"]
        energy_per_protein = model_info["protein"]["e_const"]
        protein_length = model_info["protein"]["length"]
        model = CombinedMomentModel(transcription_rate, rna_degradation_rate, energy_per_nucleotide, energy_per_rna,
                                    rna_length,
                                    translation_rate, protein_degradation_rate, energy_per_amino_acid,
                                    energy_per_protein, protein_length)

        return model


class TranscriptionfactorInputs(Protein):
    def __init__(self, gate_lib_entry_dict: dict,
                 gate_lib: GateLibCollectionBased = None):
        self.collection_identifier = "tf_inputs"
        super().__init__(collection_identifier=self.collection_identifier,
                         gate_lib_entry_dict=gate_lib_entry_dict,
                         gate_lib=gate_lib)

        # A single construct can be asssociated to multiple sequences (for example codon alternatives)
        self.LUT = {float(key): self.model_info["LUT"][key] for key in self.model_info["LUT"]}
        self.signal_name = "signal_" + self.name

    def _populate_model(self):
        def model(cell_state, sim_settings=None):
            output_val = None
            name = self.name
            signal_name = self.signal_name
            if signal_name in cell_state:
                val = cell_state[signal_name]
                output_val = self.LUT[val]
                cell_state[name] = output_val
                cell_state[name + "_rpu"] = output_val
            return output_val

        return model


# class RNADynamics(GateLibModelEntry):
#     def __init__(self, gate_lib_entry_dict):
#         self.collection_identifier = "rna_dynamics"
#         super().__init__(collection_identifier=self.collection_identifier, gate_lib_entry_dict=gate_lib_entry_dict)
#
#     def _populate_model(self):
#         model_info = self.model_info
#         # Includes molecule dependent transcription, RNA degradation and the respective energy requirements and lengths.
#
#         transcription_rate = model_info["transcription_rate"]
#         degradation_rate = model_info["degradation_rate"]
#         energy_per_nucleotide = model_info["energy_per_nucleotide"]
#         energy_per_rna = model_info["energy_per_rna"]
#         length = model_info["length"]
#         model = RNAMomentModel(transcription_rate, degradation_rate,
#                                energy_per_nucleotide, energy_per_rna,
#                                length)
#         return model
#
#
# class ProteinDynamics(GateLibModelEntry):
#     def __init__(self, gate_lib_entry_dict):
#         self.collection_identifier = "protein_dynamics"
#         super().__init__(collection_identifier=self.collection_identifier, gate_lib_entry_dict=gate_lib_entry_dict)
#
#     def _populate_model(self):
#         model_info = self.model_info
#         # Includes molecule dependent translation, protein degradation and the respective energy requirements and lengths.
#
#         translation_rate = model_info["translation_rate"]
#         degradation_rate = model_info["degradation_rate"]
#         energy_per_amino_acid = model_info["energy_per_amino_acid"]
#         energy_per_protein = model_info["energy_per_protein"]
#         length = model_info["length"]
#         model = ProteinMomentModel(translation_rate, degradation_rate,
#                                    energy_per_amino_acid, energy_per_protein,
#                                    length)
#         return model


# class Protein(GateLibModelEntry):
#     def __init__(self, gate_lib_entry_dict):
#         self.collection_identifier = "proteins"
#         super().__init__(collection_identifier=self.collection_identifier, gate_lib_entry_dict=gate_lib_entry_dict)
#
#     def _populate_model(self):
#         raise Exception("Not Implemented Yet")


class Sequence(GateLibEntry):
    def __init__(self, gate_lib_entry_dict: dict,
                 gate_lib: GateLibCollectionBased = None):
        self.collection_identifier = "sequences"
        super().__init__(collection_identifier=self.collection_identifier,
                         gate_lib_entry_dict=gate_lib_entry_dict,
                         gate_lib=gate_lib)

        self.type = gate_lib_entry_dict["type"]
        self.sequence = gate_lib_entry_dict["dna_sequence"]


class UTR(GateLibEntry):
    def __init__(self, gate_lib_entry_dict: dict,
                 gate_lib: GateLibCollectionBased = None):
        self.collection_identifier = "utrs"
        super().__init__(collection_identifier=self.collection_identifier,
                         gate_lib_entry_dict=gate_lib_entry_dict,
                         gate_lib=gate_lib)
        self.sequence_id = gate_lib_entry_dict["sequence_id"]


class Terminator(GateLibEntry):
    def __init__(self, gate_lib_entry_dict: dict,
                 gate_lib: GateLibCollectionBased = None):
        self.collection_identifier = "terminators"
        super().__init__(collection_identifier=self.collection_identifier,
                         gate_lib_entry_dict=gate_lib_entry_dict,
                         gate_lib=gate_lib)
        self.sequence_id = gate_lib_entry_dict["sequence_id"]


class Device(GateLibEntry):
    """
    Represents the "Genetic Gate" in the meaning of Nielsen et al. 2016

    Needs to be populated last.
    """

    def __init__(self, gate_lib_entry_dict: dict,
                 gate_lib: GateLibCollectionBased = None):
        self.collection_identifier = "devices"
        super().__init__(collection_identifier=self.collection_identifier,
                         gate_lib_entry_dict=gate_lib_entry_dict,
                         gate_lib=gate_lib)

        self.group = gate_lib_entry_dict["group"]
        self.primitive_identifier = gate_lib_entry_dict["primitive_identifier"]
        self.promoter_id = gate_lib_entry_dict["promoter_id"]
        self.utr_id = gate_lib_entry_dict["utr_id"]
        self.cds_id = gate_lib_entry_dict["cds_id"]
        self.terminator_id = gate_lib_entry_dict["terminator_id"]
        self.input_id = None
        self.ids = [self.promoter_id, self.utr_id, self.cds_id, self.terminator_id]
        self.promoter = None
        self.utr = None
        self.cds = None
        self.terminator = None

        if gate_lib is not None:
            def callback(gate_lib, *args, **kwargs):
                self.promoter = gate_lib(self.promoter_id, Promoter)
                self.utr = gate_lib(self.utr_id, UTR)
                self.cds = gate_lib(self.cds_id, CodingSequence)
                self.terminator = gate_lib(self.terminator_id, Terminator)

            gate_lib.add_on_completion_callback(callback)
        # if parts is not None:
        #     parts_to_use = dict(parts)
        #     parts_to_use = {key: parts_to_use[key] if key in parts_to_use else None for key in self.ids}
        #
        #     self.promoter = parts_to_use[self.promoter_id]
        #     self.utr = parts_to_use[self.utr_id]
        #     self.cds = parts_to_use[self.cds_id]
        #     self.terminator_id = parts_to_use[self.terminator_id]
        # else:
        #     raise Exception("Implementation not completed")

    def __str__(self):
        return "Device: " + ", ".join([elem if elem is not None else "-" for elem in self.ids])


if __name__ == '__main__':
    # json_file_old = JsonFile(path="ARCTICsim/simulator_nonequilibrium/data/gate_libs/gate_lib_yeast.json")
    # gate_lib_old = GateLib(json_file=json_file_old)
    json_file = JsonFile(path="ARCTICsim/simulator_nonequilibrium/data/gate_libs/gate_lib_draft.json")
    gate_lib = GateLibCollectionBased(json_file=json_file)

    promoter = gate_lib.objects_by_collection["promoters"][0]
    promoter_constitutive = gate_lib("promoter_pConstitutive")
    promoter = promoter_constitutive
    proteins = gate_lib.objects_by_collection["proteins"][0]
    result = promoter({"c": 1}, {})

    sim_settings = {"mode": "samp"}
    result = gate_lib.objects_by_id["sensor_promoter_Ptet"]({"c": 5.0}, sim_settings=sim_settings)
    prot_result = proteins(result, sim_settings=sim_settings)

    num_samples = 8 * 10 ** 4
    input_vals = 1000.0 * np.random.random(num_samples)
    profiler = cProfile.Profile()
    profiler.enable()

    for i in range(num_samples):
        result = promoter({"c": input_vals[i]}, {})
        protein_result = proteins(result, sim_settings={"mode": "samp"})

    profiler.disable()
    profiler.dump_stats("profile_results_new.prof")
    profiler.print_stats()

    # protein_result = proteins(result, sim_settings={"mode": "samp"})
    pass
    print("\n\n\nOld Implementation")
    gate_lib_old
    not_1 = gate_lib_old.gates_by_type_and_name["NOT"]['P1_PhlF']

    profiler = cProfile.Profile()
    profiler.enable()

    for i in range(num_samples):
        not_1(input_vals[i], sim_settings={"mode": "samp"})

    profiler.disable()
    profiler.dump_stats("profile_results_old.prof")
    profiler.print_stats()
    #
    # not_1 = gate_lib.gates[1]
    # not_2 = gate_lib.gates[3]
    #
    # plt.figure()
    # for not_i in [not_1, not_2]:
    #     samples = [not_i(10000000) for _ in range(1000)]
    #     plt.hist(samples, alpha=0.5, bins=100)
    #
    # plt.show()
    # pass
