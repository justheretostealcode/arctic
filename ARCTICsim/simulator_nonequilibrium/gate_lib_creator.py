"""
Exemplary script to derive a Non-Equilibrium Compatible Library from the Cello for Yeast Library
"""
import json

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

from simulator.circuit import Gene
from simulator.gatelib import Promoter, UTR, Protein, TranscriptionfactorInputs, \
    SensorPromoter
from simulator.histogram_data import GateCytometryData
from simulator.utils import JsonFile


def filter_by_collection(data_as_list, collection_name):
    filtered_data_as_list = filter(lambda elem: elem["collection"] == collection_name, data_as_list)
    return list(filtered_data_as_list)


def get_promoter_entry(name, cognate_transcription_factors=None, sequence_ids=None, scaling_factor=1,
                       num_binding_sites=2):
    if cognate_transcription_factors is None:
        cognate_transcription_factors = []
    if sequence_ids is None:
        sequence_ids = []

    promoter_entry = {"identifier": "promoter_" + name,
                      "name": name,
                      "collection": "promoters",
                      "cognate_transcription_factors": cognate_transcription_factors,
                      "sequence_ids": sequence_ids,
                      "technology_mapping": None,
                      "model_info": None
                      }

    if num_binding_sites == 2:
        promoter_entry["model_info"] = {"N_PROMOTER_STATES": 6,
                                        "N_PARAMETERS": 16,
                                        "INPUT_SCALING_FACTOR": scaling_factor,
                                        "PROMOTER_ACTIVITY": [0, 0, 0, 1, 1, 1],
                                        "INFINITESIMAL_GENERATOR_FUNCTION": {"matrices": {"1": [[]],
                                                                                          "c": [[]]
                                                                                          }}}
    elif num_binding_sites == 3:
        promoter_entry["model_info"] = {"N_PROMOTER_STATES": 8,
                                        "N_PARAMETERS": 22,
                                        "INPUT_SCALING_FACTOR": scaling_factor,
                                        "PROMOTER_ACTIVITY": [0, 0, 0, 0, 1, 1, 1, 1],
                                        "INFINITESIMAL_GENERATOR_FUNCTION": {"matrices": {"1": [[]],
                                                                                          "c": [[]]
                                                                                          }}}

    return promoter_entry


def get_protein_entry(name, sequence_ids=None,
                      transcription_rate=None, rna_degradation_rate=None, rna_length=None,
                      translation_rate=None, protein_degradation_rate=None, protein_length=None):
    if sequence_ids is None:
        sequence_ids = []
    protein_entry = {"identifier": "protein_" + name,
                     "name": name,
                     "collection": "proteins",
                     "sequence_ids": sequence_ids,
                     "model_info": {"rna": {"transcription_rate": transcription_rate,
                                            "degradation_rate": rna_degradation_rate,
                                            "e": 16,
                                            "e_const": 0,
                                            "length": rna_length
                                            },
                                    "protein": {"translation_rate": translation_rate,
                                                "degradation_rate": protein_degradation_rate,
                                                "e": 42,
                                                "e_const": 52,
                                                "length": protein_length
                                                }
                                    }
                     }
    return protein_entry


def get_device_entry(name, group, regulator=None, primitive_identifer=None, utr_id=None,
                     cds_id=None, promoter_id=None, terminator_id=None, color=None):
    if primitive_identifer is None:
        primitive_identifer = []

    device_entry = {"collection": "devices",
                    "identifier": "device_" + name,
                    "name": name,
                    "group": group,
                    "regulator": regulator,
                    "primitive_identifier": primitive_identifer,
                    "utr_id": utr_id,  # ToDo Add the corresponding ID
                    "cds_id": cds_id,  # ToDo Add the corresponding ID
                    "terminator_id": terminator_id,  # terminators are currently not included in the library
                    "promoter_id": promoter_id,
                    "color": color
                    }  # ToDo Add the corresponding ID
    return device_entry


def get_sequence_entry(name, sequence=""):
    sequence_entry = {"collection": "sequences",
                      "identifier": "sequence_" + name,
                      "name": name,
                      "sequence": sequence}
    return sequence_entry


def add_outputs(custom_lib, reporters, rates: dict):
    transcription_rate = rates["transcription_rate"]
    translation_rate = rates["translation_rate"]
    rna_degradation_rate = rates["rna_degradation_rate"]
    protein_degradation_rate = rates["protein_degradation_rate"]
    cds_length = 561
    rna_length = 600
    protein_length = cds_length / 3
    for reporter in reporters:
        utr_entry = {"collection": "utrs",
                     "identifier": "utr_" + reporter["utr_name"],
                     "name": reporter["utr_name"],
                     "sequence_id": None}

        protein_entry = get_protein_entry(name=reporter["name"],
                                          sequence_ids=[],
                                          transcription_rate=transcription_rate,
                                          rna_degradation_rate=rna_degradation_rate,
                                          rna_length=rna_length,
                                          translation_rate=translation_rate,
                                          protein_degradation_rate=protein_degradation_rate,
                                          protein_length=protein_length)

        device_entry = get_device_entry(name=reporter["name"], group=reporter["name"],
                                        primitive_identifer=["OUTPUT_OR2", "OUTPUT_BUFFER"],
                                        utr_id=utr_entry["identifier"],
                                        cds_id=protein_entry["identifier"])

        custom_lib.append(utr_entry)
        custom_lib.append(protein_entry)
        custom_lib.append(device_entry)


def add_inputs(custom_lib, tf_inputs, reporter_information):
    # {"group": "aTc", "promoter": "ptet", "protein": "tetR", "low": 0.002, "high": 2.5},
    reference_sensor_promoter_entry = None
    for tf_input in tf_inputs:
        # protein_entry = {"identifier": "tf_input_" + tf_input["protein"],
        #                  "name": tf_input["protein"],
        #                  "collection": "tf_inputs",
        #                  "sequence_ids": [],
        #                  "model_info": {
        #                      "LUT": {
        #                          "0": 10,
        #                          "1": 0.001
        #                      }}}
        # ToDo match Promoter
        promoter_entry = get_promoter_entry(name=tf_input["promoter"],
                                            cognate_transcription_factors=[tf_input["group"], tf_input["protein"]],
                                            sequence_ids=None)
        promoter_entry["collection"] = "sensor_promoters"
        promoter_entry["identifier"] = "sensor_" + promoter_entry["identifier"]
        data = tf_input["data"]
        promoter_entry["model_info"] = {"low": tf_input["low"],
                                        "high": tf_input["high"],
                                        "LUT": data}
        promoter_entry["technology_mapping"] = {"y_min": data[tf_input["low"]],
                                                "y_max": data[tf_input["high"]]}

        # response_characteristic = {"X": np.array(list(data.keys())),
        #                            "Y": np.array([[data[inducer_concentration],
        #                                            0.25 * data[inducer_concentration]
        #                                            ] for
        #                                           inducer_concentration in data]),
        #                            "labels": ["mean",
        #                                       "variance"
        #                                       ]}
        # input_information = {"tf_input": protein_entry,
        #                      "promoter": promoter_entry}
        # promoter_entry = match_promoter(response_characteristic=response_characteristic,
        #                                 reporter_information=reporter_information,
        #                                 input_information=input_information,
        #                                 match_input_sensor=True)

        device_entry = get_device_entry(name=tf_input["group"], group=tf_input["group"], primitive_identifer=["INPUT"],
                                        promoter_id=promoter_entry["identifier"],
                                        # cds_id=protein_entry["identifier"]
                                        )

        if tf_input["group"] == "IPTG":
            reference_sensor_promoter_entry = promoter_entry

        # custom_lib.append(protein_entry)
        custom_lib.append(promoter_entry)
        custom_lib.append(device_entry)
        pass
    return reference_sensor_promoter_entry


def get_inflection_point(func, f_val=None):
    def derivative(f, x_param):
        X = np.array([0.999, 1.001]) * x_param
        y = np.empty(shape=X.shape)
        for iX, x in enumerate(X):
            y[iX] = f(x)

        y_diff = np.diff(y)
        x_diff = np.diff(X)
        deriv = y_diff / x_diff
        return deriv[0]

    def find_position_of_steepest_descend(f, bounds, min_dist=0.001, use_log=False):

        while bounds[1] - bounds[0] >= min_dist:
            a, b = bounds
            assert a <= b
            center = (a + b) / 2
            if use_log:
                center = np.sqrt(a * b)

            # ax.scatter(center, f(center), marker="x")

            X = np.array([a, center, b], dtype=float)
            Y = np.array([f(x) for x in X], dtype=float)
            X_log = X
            Y_log = Y

            if use_log:
                X_log = np.log10(X)
                Y_log = np.log10(Y)

            Y_diff = np.diff(Y_log)
            X_diff = np.diff(X_log)
            deriv = Y_diff / X_diff
            deriv = np.abs(deriv)

            if deriv[1] > deriv[0]:
                bounds = X[1:]
            else:
                bounds = X[:2]

        pos = center
        return pos

    def find_position_of_steepest_descend(f, bounds, min_dist=0.001, use_log=False):

        while bounds[1] - bounds[0] >= min_dist:
            a, b = bounds
            assert a <= b
            center = (a + b) / 2
            if use_log:
                center = np.sqrt(a * b)

            X = np.array([a, center, b], dtype=float)
            Y = np.array([f(x) for x in X], dtype=float)
            X_log = X
            Y_log = Y

            if use_log:
                X_log = np.log10(X)
                Y_log = np.log10(Y)

            Y_diff = np.diff(Y_log)
            X_diff = np.diff(X_log)
            deriv = Y_diff / X_diff
            deriv = np.abs(deriv)

            if deriv[1] > deriv[0]:
                bounds = X[1:]
            else:
                bounds = X[:2]

        pos = center
        return pos

    def find_position_of_func_val(f, f_val, bounds, min_dist=0.001, use_log=False):

        while bounds[1] - bounds[0] >= min_dist:
            a, b = bounds
            assert a <= b
            center = (a + b) / 2
            if use_log:
                center = np.sqrt(a * b)

            X = np.array([center], dtype=float)
            Y = np.array([f(x) for x in X], dtype=float)

            if Y[0] < f_val:
                bounds = (a, center)
            else:
                bounds = (center, b)

        pos = center
        return pos

    f = func
    if f_val is None:
        x_m = find_position_of_steepest_descend(f, bounds=[1.0 * 10 ** -4, 10 ** 2], min_dist=0.001)
    else:
        x_m = find_position_of_func_val(f, f_val=f_val, bounds=[1.0 * 10 ** -4, 10 ** 2], min_dist=0.001, use_log=True)
    y_m = f(x_m)
    d_y_m = derivative(f, x_m)

    # print("Ratio:", f(10) / y_m)
    inflection_point_info = {"inflection_point": {"x_m": x_m, "y_m": y_m, "d_y_m": d_y_m}}
    return inflection_point_info


def match_promoter(response_characteristic: dict,
                   reporter_information: dict,
                   gate_information: dict = None,
                   input_information: dict = None, match_input_sensor=False):
    """
    This method performs the parameter matching for the promoter model to the response characteristics

    :param response_characteristic: dict providing the input values and corresponding outputcharacteristics
    :return:
    """

    utr_reporter = UTR(reporter_information["cds"]) if "utr" in reporter_information else None
    protein_reporter = Protein(reporter_information["cds"])
    terminator_reporter = None
    reporter_name = protein_reporter.name + "_rpu"
    # Steps
    # Setup Model to train (Gene includes YFP output and the input is a TF Level (possibly derived from RPU))
    # Match model to training data
    # if not match_input_sensor:
    gate_name = None
    if input_information is None:
        raise Exception("Input Information is essentially required for parameter matching.")
    else:
        # tf_input = TranscriptionfactorInputs(input_information["tf_input"])
        promoter_sensor = SensorPromoter(input_information["promoter"])
        input_signal_name = promoter_sensor.cognate_transcription_factors[0]

    genes = []
    if gate_information is None:
        sensor_gene = Gene(id="Sensor", promoter=promoter_sensor, utr=utr_reporter, cds=protein_reporter)
        genes.append(sensor_gene)
        promoter = promoter_sensor
    else:
        gate_name = gate_information["name"]
        utr_gate = UTR(gate_information["utr"]) if "utr" in gate_information else None
        protein_gate = Protein(gate_information["cds"])
        terminator_gate = None
        promoter_gate = Promoter(gate_information["promoter"])

        sensor_gene = Gene(id="Sensor", promoter=promoter_sensor, utr=utr_gate, cds=protein_gate)
        reporter_gene = Gene(id="Reporter", promoter=promoter_gate, utr=utr_reporter, cds=protein_reporter)

        promoter = promoter_gate

        genes.append(sensor_gene)
        genes.append(reporter_gene)

    X = response_characteristic["X"]
    X_rpu = response_characteristic["X_rpu"]
    Y = response_characteristic["Y"]
    # X[0] = 10**(-2)
    labels = response_characteristic["labels"]

    # y = 0.05
    # s = np.max(X) * 0.05
    # k_m = k_m = s * y / (1 - y)
    # ligand_to_tf_level = lambda x: 1 / (1 + (x / k_m) ** 1)

    # Case distinction
    # Case 1: Parameter estimation of a genetic gate -> both genes are used and matched input sensors and existing reporters are required
    # Case 2: Parameter estimation of an input sensor -> only the reporter gene is required. The input sensor's promoter is directly attached to it.

    def model(plasmid_input_vals, sim_settings):
        n_samples = sim_settings["n_samples_simulation"]
        mode = sim_settings["mode"]
        # output_vals = np.empty(shape=sample_count)

        cell_state = {"energy": np.zeros(n_samples)}
        cell_state.update(plasmid_input_vals)
        # sim_settings["mode"] = "samp" if mode == "det-var" else mode
        for gene in genes:
            output_dict = gene(cell_state, sim_settings=sim_settings)

        output_vals = cell_state[reporter_name]
        # gene.gene_state
        sim_settings["mode"] = mode
        scaling_factor = gene.cds.model.scaling_factor
        mean_prot = gene.gene_state["protein_mean"]
        var_prot = gene.gene_state["protein_var"]
        mean_out = mean_prot / scaling_factor
        var_out = var_prot / scaling_factor ** 2

        if mode == "det" or mode == "det-var":
            return np.mean(mean_out), np.mean(var_out)
        else:
            return np.mean(output_vals), np.var(output_vals)

    def update_model(params):
        # new_params = params
        new_params = 10 ** params
        new_params[:2] = params[:2]
        promoter.model.insert_params(new_params)

    def loss_func(Y, Y_pred, custom_weights=None):
        weights = np.ones(shape=Y.shape)
        if custom_weights is not None:
            weights = custom_weights
        weights = weights / np.sum(weights)

        loss = 0.0
        # loss += np.sum((weights * np.power(np.log(Y) - np.log(Y_pred), 2))[Y_pred != 0])
        loss += np.sum((weights[:, 0] * np.power(np.log(Y[:, 0]) - np.log(Y_pred[:, 0]), 2)))

        if np.any(weights[:, 1] != 0):
            loss += np.sum((weights[:, 1] * np.power(np.log(np.sqrt(Y[:, 1])) - np.log(np.sqrt(Y_pred[:, 1])), 2)))
            # loss += np.sum((weights[:, 0] * np.power(np.log(np.sqrt(Y[:, 1])) - np.log(np.sqrt(Y_pred[:, 0])), 2)))

        return loss.item()

    def error_func(params, training_data):
        X = training_data["X"]
        Y = training_data["Y"]
        n_samples = training_data["n_samples_simulation"]
        matching_mode = training_data["mode"]
        update_model(params)

        sim_settings = {"mode": matching_mode, "n_samples_simulation": sample_count}
        # X = response_characteristic["X"]
        # Y = response_characteristic["Y"]
        # labels = response_characteristic["labels"]
        Y_pred = [None] * len(X)
        input_vals_dict = {}
        for iX, x in enumerate(X):
            input_vals_dict[input_signal_name] = np.ones(n_samples) * x
            y_pred = model(plasmid_input_vals=input_vals_dict, sim_settings=sim_settings)
            y_pred = {label: y_pred[iL] for iL, label in enumerate(["mean", "variance"])}
            Y_pred[iX] = [y_pred[label] for label in labels]

        Y_pred = np.array(Y_pred)
        custom_weights = np.ones(shape=Y_pred.shape)  # Equals average weighting.
        # custom_weights = np.zeros(shape=Y_pred.shape)
        custom_weights[0, 0] = 10
        custom_weights[1, 0] = 4
        # custom_weights[2, 0] = 2
        # custom_weights[-2, 0] = 2
        custom_weights[-2, 0] = 4
        custom_weights[-1, 0] = 10
        custom_weights[:, 1] = 0.2
        custom_weights[0, 1] = 5
        custom_weights[1, 1] = 2
        custom_weights[-2, 1] = 2
        custom_weights[-1, 1] = 5
        if matching_mode == "det":
            custom_weights[:, 1] = 0
        # custom_weights[:, 1] = custom_weights[:, 0] / 10
        # Can realize less weighting of variance results.
        loss = loss_func(Y, Y_pred, custom_weights=custom_weights)
        # deviation = deviation_func(params)
        # loss = deviation

        ######################
        # Score Monotonicity #
        ######################
        # cognate_tf = promoter_gate.cognate_transcription_factors[0]

        n_vals = 50
        output_dict = promoter_gate(
            in_vals={"c": np.logspace(-4, 2, n_vals) / promoter_gate.model.input_scaling_factor},
            sim_settings={"mode": "samp", "n_samples_simulation": n_vals})
        average_promoter_activity = output_dict["average_promoter_activity"]
        diffs = np.diff(np.log(average_promoter_activity))
        monotonicity_loss = np.sum(np.abs(diffs[diffs > 0]))

        # genes

        error = loss + 0.1 * monotonicity_loss
        return error

    #
    # ref_params = [3.589279542832028, 0.003109276598567825, 479.0328585600103, 5.579281617098493e-11,
    #               5.544387725272706e-11, 5.7219952157354176e-11, 5.581332098763365e-11, 6.669774560347158e-11,
    #               1167.502391260739, 4834.371783185657, 2.1532840435156686e-06, 92474.44318734983]
    #
    # def deviation_func(params):
    #     y_on_ref = ref_params[0]
    #     y_off_ref = ref_params[1]
    #
    #     alphas_ref = ref_params[2:2 + 5]
    #     betas_ref = ref_params[2 + 5:]
    #     y_on, y_off, k_01, k_03, k_10, k_12, k_14, k_21, k_25, k_30, k_34, k_41, k_43, k_45, k_52, k_54 = params
    #     alphas = [
    #         k_03 * k_10 * k_21 * k_41 * k_52 + k_03 * k_10 * k_21 * k_43 * k_52 + k_03 * k_14 * k_21 * k_43 * k_52 + k_03 * k_10 * k_21 * k_41 * k_54 + k_03 * k_10 * k_25 * k_41 * k_54 + k_03 * k_10 * k_21 * k_43 * k_54 + k_03 * k_14 * k_21 * k_43 * k_54 + k_03 * k_10 * k_25 * k_43 * k_54 + k_03 * k_14 * k_25 * k_43 * k_54,
    #         k_01 * k_14 * k_21 * k_30 * k_52 + k_03 * k_10 * k_21 * k_34 * k_52 + k_03 * k_14 * k_21 * k_34 * k_52 + k_01 * k_14 * k_21 * k_43 * k_52 + k_03 * k_10 * k_21 * k_45 * k_52 + k_01 * k_14 * k_21 * k_30 * k_54 + k_01 * k_14 * k_25 * k_30 * k_54 + k_03 * k_10 * k_21 * k_34 * k_54 + k_03 * k_14 * k_21 * k_34 * k_54 + k_03 * k_10 * k_25 * k_34 * k_54 + k_03 * k_14 * k_25 * k_34 * k_54 + k_01 * k_14 * k_21 * k_43 * k_54 + k_03 * k_12 * k_25 * k_43 * k_54 + k_01 * k_14 * k_25 * k_43 * k_54,
    #         k_01 * k_12 * k_25 * k_30 * k_41 + k_03 * k_12 * k_25 * k_34 * k_41 + k_01 * k_12 * k_25 * k_30 * k_43 + k_01 * k_14 * k_21 * k_30 * k_45 + k_01 * k_14 * k_25 * k_30 * k_45 + k_03 * k_10 * k_21 * k_34 * k_45 + k_03 * k_14 * k_21 * k_34 * k_45 + k_03 * k_10 * k_25 * k_34 * k_45 + k_03 * k_14 * k_25 * k_34 * k_45 + k_01 * k_14 * k_21 * k_34 * k_52 + k_01 * k_12 * k_25 * k_30 * k_54 + k_01 * k_14 * k_21 * k_34 * k_54 + k_03 * k_12 * k_25 * k_34 * k_54 + k_01 * k_14 * k_25 * k_34 * k_54 + k_01 * k_12 * k_25 * k_43 * k_54,
    #         k_01 * k_12 * k_25 * k_34 * k_41 + k_01 * k_12 * k_25 * k_30 * k_45 + k_01 * k_14 * k_21 * k_34 * k_45 + k_03 * k_12 * k_25 * k_34 * k_45 + k_01 * k_14 * k_25 * k_34 * k_45 + k_01 * k_12 * k_25 * k_34 * k_54,
    #         k_01 * k_12 * k_25 * k_34 * k_45]
    #     alphas = np.array(alphas)
    #     betas = [
    #         k_10 * k_21 * k_30 * k_41 * k_52 + k_10 * k_21 * k_30 * k_43 * k_52 + k_14 * k_21 * k_30 * k_43 * k_52 + k_10 * k_21 * k_30 * k_41 * k_54 + k_10 * k_25 * k_30 * k_41 * k_54 + k_10 * k_21 * k_30 * k_43 * k_54 + k_14 * k_21 * k_30 * k_43 * k_54 + k_10 * k_25 * k_30 * k_43 * k_54 + k_14 * k_25 * k_30 * k_43 * k_54,
    #         k_01 * k_21 * k_30 * k_41 * k_52 + k_03 * k_21 * k_34 * k_41 * k_52 + k_10 * k_21 * k_34 * k_41 * k_52 + k_01 * k_21 * k_30 * k_43 * k_52 + k_10 * k_21 * k_30 * k_45 * k_52 + k_01 * k_21 * k_30 * k_41 * k_54 + k_01 * k_25 * k_30 * k_41 * k_54 + k_03 * k_21 * k_34 * k_41 * k_54 + k_10 * k_21 * k_34 * k_41 * k_54 + k_03 * k_25 * k_34 * k_41 * k_54 + k_10 * k_25 * k_34 * k_41 * k_54 + k_01 * k_21 * k_30 * k_43 * k_54 + k_01 * k_25 * k_30 * k_43 * k_54 + k_12 * k_25 * k_30 * k_43 * k_54,
    #         k_01 * k_12 * k_30 * k_41 * k_52 + k_03 * k_12 * k_34 * k_41 * k_52 + k_01 * k_21 * k_34 * k_41 * k_52 + k_01 * k_12 * k_30 * k_43 * k_52 + k_01 * k_14 * k_30 * k_45 * k_52 + k_01 * k_21 * k_30 * k_45 * k_52 + k_03 * k_10 * k_34 * k_45 * k_52 + k_03 * k_14 * k_34 * k_45 * k_52 + k_03 * k_21 * k_34 * k_45 * k_52 + k_10 * k_21 * k_34 * k_45 * k_52 + k_01 * k_12 * k_30 * k_41 * k_54 + k_03 * k_12 * k_34 * k_41 * k_54 + k_01 * k_21 * k_34 * k_41 * k_54 + k_01 * k_25 * k_34 * k_41 * k_54 + k_01 * k_12 * k_30 * k_43 * k_54,
    #         k_01 * k_12 * k_34 * k_41 * k_52 + k_01 * k_12 * k_30 * k_45 * k_52 + k_03 * k_12 * k_34 * k_45 * k_52 + k_01 * k_14 * k_34 * k_45 * k_52 + k_01 * k_21 * k_34 * k_45 * k_52 + k_01 * k_12 * k_34 * k_41 * k_54,
    #         k_01 * k_12 * k_34 * k_45 * k_52]
    #     betas = np.array(betas)
    #     divisor = len(alphas) + len(betas)
    #     deviation = np.sum(np.power(alphas - alphas_ref, 2))
    #     deviation += np.sum(np.power(betas - betas_ref, 2))
    #     deviation /= divisor
    #     return deviation

    # matching_modes = ["det", "samp"]
    # sample_counts = [1, 100]
    # matching_modes = ["det", "det-var"]
    # sample_counts = [1, 1]

    if True:
        # matching_modes = ["det", "det-var"]
        # sample_counts = [1, 1]  # "det-var" only requires a single sample

        matching_modes = ["det-var"]
        sample_counts = [1]  # "det-var" only requires a single sample
    else:
        matching_modes = ["samp"]
        sample_counts = [200]
    max_fev_per_params = [100, 500]  # 1200
    max_fev_per_params = [500]  # 1200
    eval_sample_count = 1000  # 1000
    num_trials = 10  # 10

    num_params = promoter_gate.model.num_trainable_parameters

    best_fun = np.infty
    best_params = None

    for iR in range(num_trials):
        mask_mean = [label == "mean" for label in labels]
        Y_mean = [y[mask_mean] for y in Y]
        y_on = np.max(Y_mean)
        y_off = np.min(Y_mean)

        # y_on, y_off, k_01, k_04, k_10, k_12, k_15, k_21, k_23, k_26, k_32, k_37, k_40, k_45, k_51, k_54, k_56, k_62, k_65, k_67, k_73, k_76

        initial_params = None

        initial_params = [y_on, y_off]
        initial_params += list(np.random.rand(num_params - 2) * 2 - 1)
        initial_params[3] = 2  # k_04
        initial_params[12] = -2  # k_40
        initial_params[6] = 1  # k_15
        initial_params[14] = -1  # k_51
        initial_params[9] = -1  # k_26
        initial_params[17] = 1  # k_62
        initial_params[11] = -2  # k_37
        initial_params[20] = 2  # k_73

        ####################################
        # Perform multiple sequential optimization steps which are combined  #
        ####################################
        for matching_mode, sample_count, max_fev_per_param in zip(matching_modes, sample_counts, max_fev_per_params):
            # The optimization routine adapts the rates in the infinitesimal generator and the promoter activity

            ####################################
            # Setup Parameter Matching Context #
            ####################################

            # # Imply inhibitory behaviour via bounds.
            bounds = [(y_on, y_on * 2), (y_off * 0.5, y_off)] + [[-5, 5] for _ in range(num_params - 2)]
            # training_data = [np.array([X[0], X[-1]]), np.array([Y[0], Y[-1]])]
            training_data = {"X": X,
                             "Y": Y,
                             "n_samples": sample_count,
                             "n_samples_simulation": sample_count,
                             "mode": matching_mode}

            # initial_params are not None when they  are set to the params obtained in the previous run in the inner loop
            if initial_params is None:
                initial_params = [y_on, y_off]
                initial_params += list(np.random.rand(num_params - 2) * 2 - 1)
                initial_params = np.array(initial_params)

            ##############################
            # Perform Parameter Matching #
            ##############################

            x0 = initial_params
            result = minimize(error_func, x0=x0,
                              args=training_data,
                              bounds=bounds,
                              method="powell",
                              # method="nelder-mead",
                              # method="TNC",
                              #   method="SLSQP",
                              tol=10 ** (-10),
                              options={"disp": True,
                                       "ftol": 10 ** (-7),
                                       # "maxfev": len(x0) * 200 if matching_mode == "samp" else len(x0) * 1000
                                       "maxfev": len(x0) * max_fev_per_param
                                       # "maxfev": len(x0) * 500 if matching_mode == "samp" else len(x0) * 1
                                       }
                              )
            new_params = result.x
            fun = result.fun
            print(f"Run {iR}: {result.fun} (y_on={y_on}, y_off={y_off})")

            initial_params = new_params

            #############################################
            # Score distribution match based on samples #
            #############################################
            matching_mode = "samp"
            sample_count = eval_sample_count
            training_data = {"X": X,
                             "Y": Y,
                             "n_samples": sample_count,
                             "n_samples_simulation": sample_count,
                             "mode": matching_mode}

            distribution_losses = [error_func(new_params, training_data) for _ in range(10)]
            distribution_loss = np.mean(distribution_losses)
            print(f"Error for current estimate: {distribution_loss} ({distribution_losses}")

            # Y_pred = [model({input_signal_name: np.ones(sample_count) * x}, sim_settings=sim_settings) for x in X]
            # Y_pred = np.array(Y_pred)

            # plt.figure()
            # ax = plt.gca()
            # ax.plot(X, Y[:, 0], "--k", label="Mean")
            # if Y.shape[1] == 2:
            #     ax.fill_between(X,
            #                     Y[:, 0] - np.sqrt(Y[:, 1]), Y[:, 0] + np.sqrt(Y[:, 1]),
            #                     color="yellow", alpha=0.1)
            #
            # ax.plot(X, Y_pred[:, 0], label="Mean Pred")
            # if Y_pred.shape[1] == 2:
            #     ax.fill_between(X,
            #                     Y_pred[:, 0] - np.sqrt(Y_pred[:, 1]), Y_pred[:, 0] + np.sqrt(Y_pred[:, 1]),
            #                     color="blue", alpha=0.1)
            # ax.set_xscale("log")
            # ax.set_yscale("log")
            # plt.show()

        # Select the run with the minimal loss after all matching steps
        if distribution_loss < best_fun:  # fun < best_fun:
            # best_fun = fun
            best_fun = distribution_loss
            best_params = new_params

    print(f"Best Run with error: {best_fun}")

    #################################
    # Parameter Matching Evaluation #
    #################################

    update_model(best_params)
    matching_mode = "samp"
    sample_count = eval_sample_count
    sim_settings = {"n_samples_simulation": sample_count, "mode": matching_mode}
    Y_pred = [model({input_signal_name: np.ones(sample_count) * x}, sim_settings=sim_settings) for x in X]
    Y_pred = np.array(Y_pred)

    plt.figure()
    ax = plt.gca()
    ax.plot(X_rpu, Y[:, 0], "--k", label="Mean")
    if Y.shape[1] == 2:
        ax.fill_between(X_rpu,
                        Y[:, 0] - np.sqrt(Y[:, 1]), Y[:, 0] + np.sqrt(Y[:, 1]),
                        color="yellow", alpha=0.1)

    ax.plot(X_rpu, Y_pred[:, 0], label="Mean Pred")
    if Y_pred.shape[1] == 2:
        ax.fill_between(X_rpu,
                        Y_pred[:, 0] - np.sqrt(Y_pred[:, 1]), Y_pred[:, 0] + np.sqrt(Y_pred[:, 1]),
                        color="blue", alpha=0.1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.set_xlim((10 ** (-2), 10 ** (-1)))
    # ax.set_ylim((10 ** (0), 10 ** (1)))
    dir_path = "data/gate_libs/figures/"
    plt.savefig(dir_path + gate_name + ".png", dpi=600)
    # plt.show()
    plt.close()

    ####################
    # Curve evaluation #
    ####################
    X_rpu_test = np.logspace(-3, 2, 100)
    # X_rpu_test = X_rpu
    n_samples = eval_sample_count

    sim_settings = {"n_samples_simulation": n_samples,
                    "n_samples": n_samples,
                    "mode": "samp"}
    gene = genes[-1]
    input_signal_name = gene.promoter.cognate_transcription_factors[0]
    Y_pred = [None] * len(X_rpu_test)
    input_scaling_factor = gene.promoter.model.input_scaling_factor
    scaling_factor_protein = gene.cds.model.scaling_factor
    cell_state = {"energy": np.zeros(n_samples)}
    for iX, x in enumerate(X_rpu_test):
        plasmid_input_vals = {}
        plasmid_input_vals[input_signal_name] = np.ones(n_samples) * (x / input_scaling_factor)
        cell_state.update(plasmid_input_vals)

        output_dict = gene(cell_state, sim_settings=sim_settings)
        levels = output_dict["protein_level"]
        # mean_P = output_dict["protein_mean"][0]
        # var_P = output_dict["protein_var"][0]
        mean_P = np.mean(levels)
        var_P = np.var(levels)
        mean_P /= scaling_factor_protein
        var_P /= scaling_factor_protein ** 2
        Y_pred[iX] = (mean_P, var_P)

    Y_pred = np.array(Y_pred)

    plt.figure()
    ax = plt.gca()
    ax.plot(X_rpu, Y[:, 0], "--k", label="Mean")
    if Y.shape[1] == 2:
        ax.fill_between(X_rpu,
                        Y[:, 0] - np.sqrt(Y[:, 1]), Y[:, 0] + np.sqrt(Y[:, 1]),
                        color="yellow", alpha=0.1)

    ax.plot(X_rpu_test, Y_pred[:, 0], label="Mean Pred")
    if Y_pred.shape[1] == 2:
        ax.fill_between(X_rpu_test,
                        Y_pred[:, 0] - np.sqrt(Y_pred[:, 1]), Y_pred[:, 0] + np.sqrt(Y_pred[:, 1]),
                        color="blue", alpha=0.1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    dir_path = "data/gate_libs/figures/"
    # ax.set_xlim((10 ** (-3), 10 ** (2)))
    # ax.set_ylim((10 ** (-3), 10 ** (2)))
    plt.savefig(dir_path + gate_name + "_rpu.png", dpi=600)
    # plt.show()

    y_on, y_off = best_params[:2]
    update_model(best_params)
    model_info = promoter_entry["model_info"]
    model_info["PROMOTER_ACTIVITY"] = list(promoter.model.promoter_activity)
    matrices = promoter.model.infinitesimal_generator_function["matrices"]
    model_info["INFINITESIMAL_GENERATOR_FUNCTION"] = {"matrices": {key: matrices[key].tolist()
                                                                   for key in matrices}}

    def func(in_val):
        input_scaling_factor = promoter_gate.model.input_scaling_factor
        # signal_name = promoter_gate.cognate_transcription_factors[0]
        sample_count = 1
        sim_settings = {"mode": "det", "n_samples_simulation": sample_count}
        input_val_dict = {"c": np.ones(sample_count) * in_val / input_scaling_factor}
        output_dict = promoter_gate(input_val_dict, sim_settings)
        avg_promoter_activity = output_dict["average_promoter_activity"]
        return avg_promoter_activity.item()

    f_val = (func(10 ** (-3)) + func(10 ** 2)) / 2
    technology_mapping = get_inflection_point(func=func, f_val=f_val)

    promoter_entry["technology_mapping"] = technology_mapping

    return promoter_entry


"""
These values are extracted from the paper and the data provided for the figures_mean-fit_2024-03-11 in the paper.
"""

atc_to_rpu = {0: 0.002,
              5: 0.004,
              10: 0.008,
              15: 0.013,
              20: 0.168,
              22.5: 0.445,
              25: 0.570,
              27.5: 0.668,
              30: 1.019,
              35: 1.227,
              40: 1.616,
              50: 1.963,
              100: 2.455,
              200: 2.548}
atc_to_rpu = {float(key): atc_to_rpu[key] for key in atc_to_rpu}

iptg_to_rpu = {0: 0.011,
               0.1: 0.017,
               0.25: 0.028,
               0.5: 0.049,
               1: 0.132,
               1.5: 0.224,
               2: 0.320,
               2.5: 0.421,
               3: 0.519,
               5: 0.822,
               10: 1.127,
               20: 1.297,
               30: 1.343,
               40: 1.365}
iptg_to_rpu = {float(key): iptg_to_rpu[key] for key in iptg_to_rpu}

xyl_to_rpu = {0: 0.003,
              0.1: 0.006,
              0.25: 0.025,
              0.5: 0.143,
              1: 0.480,
              1.5: 0.859,
              2: 1.133,
              3: 1.409,
              5: 1.662,
              10: 1.804,
              20: 1.904,
              30: 1.949}

xyl_to_rpu = {float(key): xyl_to_rpu[key] for key in xyl_to_rpu}

reference_inducer_inputs = np.array([0, 0.1, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 5, 10, 20])
"""
End of Paper data
"""

if __name__ == '__main__':
    # The cello for yeast library can be downloaded in
    # the Supplementary Information section of https://www.nature.com/articles/s41564-020-0757-2#additional-information
    cello_library_path = "data/reference_data/yeast/SC1C1G1T1.UCF.json"
    cello_library_json = JsonFile(path=cello_library_path)
    cello_library = cello_library_json.data
    collections = set([elem["collection"] for elem in cello_library])

    cello_models = filter_by_collection(cello_library, "models")
    cello_functions = filter_by_collection(cello_library, "functions")
    cello_structures = filter_by_collection(cello_library, "structures")
    cello_cytometry = list(filter(lambda elem: elem["name"].endswith("_cytometry"), cello_functions))
    cello_toxicity = list(filter(lambda elem: elem["name"].endswith("_toxicity"), cello_functions))
    cello_parts = filter_by_collection(cello_library, "parts")

    cello_library_by_collection = {collection: [elem for elem in cello_library if elem["collection"] == collection]
                                   for
                                   collection in collections}
    cello_structures_by_name = {elem["name"]: elem for elem in cello_structures}
    cello_parts_by_name = {elem["name"]: elem for elem in cello_parts}
    cello_cytometry_by_name = {elem["name"]: elem for elem in cello_cytometry}
    cello_toxicity_by_name = {elem["name"]: elem for elem in cello_toxicity}

    # Useful for conversion to other gatelibrary formats such as the cello E. Coli lib.
    dict_ids = {"SPECIES": "Yeast",
                "name": "name",
                "table": "table",
                "bin": "bin",
                "output": "output",
                "x": "x"}

    """
    The relevant collections are
    "gates":        Includes an overview on all the genetic gates present, their names,
                    associated structures, transcription factors and gate types.
    "models":       Includes the cello model representation of the genetic gates
    "structures":   Defines the the genetic gates with output devices (promoters) 
                    and corresponding inner signaling molecules (transcription factors). 
                    Also includes the codon variants of the promoters as well as the 
                    proteins and the kozak sequences to use.
    "functions":    The information included are input to output relations. Of particular 
                    importance is the cytometry data identifiable with the suffix 
                    "_cytometry" preced by the associated genetic gate's name. Also the 
                    characteristics of the toxicity are captured (suffix "_toxicity") and
                    equations for the "Hill_response" and "linear_input_composition" 
                    provided. 
    "parts":        Provides the DNA sequence of all components. 
                    
    Conventions     
                    Naming format of genetic gates in the cello library
                    The genetic gates are indicated by a name including an underscore, for 
                    example "P1_LexA". The part following the underscore gives rise to the 
                    transcription factor used inside the gate. With this information, the 
                    first part then determines the promoter used. In the context of the example
                    "P1" refers to "pLexA1" as it is the first variant of the promoter for the 
                    transcription factor "LexA". The second promoter variant is "pLexA2"
                    Further underscores add information on the codon variant (e.g. "pLexA1_a" or 
                    "pLexA1_b") or characterize the information provided for the genetic gate as
                    "P1_LexA_structure" identifies the structure of the genetic gate "P1_LexA",
                    while "P1_LexA_toxicity" gives rise to the corresponding toxicity information.
                    
                    
                    
    
    """
    num_binding_sites = 3

    # Our rate information
    transcription_rate = 24 / 3600
    translation_rate = 200 / 3600
    rna_degradation_rate = 5.8 * 10 ** (-4)
    protein_degradation_rate = 2.9 * 10 ** (-4)
    l_m = transcription_rate / rna_degradation_rate
    l_p = translation_rate / protein_degradation_rate
    eu_to_rpu_scaling_factor = (l_m * l_p) ** (-1)

    cello_gates = cello_library_by_collection["gates"]
    custom_lib = []

    # Reporter Output
    rates = {"transcription_rate": transcription_rate,
             "translation_rate": translation_rate,
             "rna_degradation_rate": rna_degradation_rate,
             "protein_degradation_rate": protein_degradation_rate}
    reporters = [{"name": "YFP", "utr_name": "Kozak6"}]
    add_outputs(custom_lib, reporters=reporters, rates=rates)

    # ToDo Extract Report to use for parameter estimation
    reporter_to_use = "YFP"
    reporter_protein = [elem for elem in custom_lib if elem["identifier"] == "protein_" + reporter_to_use][0]
    reporter_information = {"cds": reporter_protein}

    # Transcription Factor Inputs
    tf_inputs = [{"group": "aTc", "promoter": "Ptet", "protein": "tetR", "low": 0.0, "high": 100.0, "data": atc_to_rpu},
                 {"group": "xyl", "promoter": "Pxyl", "protein": "xylR", "low": 0.0, "high": 10.0, "data": xyl_to_rpu},
                 {"group": "IPTG", "promoter": "Plac", "protein": "lacl", "low": 0.0, "high": 20.0, "data": iptg_to_rpu}
                 ]

    reference_sensor_promoter_entry = add_inputs(custom_lib, tf_inputs=tf_inputs,
                                                 reporter_information=reporter_information)

    # Create the parameters for the actual gates
    for iG, cello_gate in enumerate(cello_gates):
        # if iG >= 1:
        #    break
        # if cello_gate["name"] != "P1_IcaR":
        #     continue

        gate_name = cello_gate["name"]
        structure = cello_structures_by_name[cello_gate["structure"]]
        raw_cytometry = cello_cytometry_by_name[gate_name + "_cytometry"]
        toxicity = cello_toxicity_by_name[gate_name + "_toxicity"]
        cytometry_data_dict = GateCytometryData.cello_cytometry_data_to_dict(raw_cytometry[dict_ids["table"]])
        cytometry = GateCytometryData(data_dict=cytometry_data_dict, gate_name=gate_name, cutoff_percentage=0.1)

        promoter_name = structure["outputs"][0].split("_")[0]
        promoter_sequence_names = structure["outputs"]
        utr_name = [elem for elem in structure["devices"] if "cassette" in elem["name"]][0]["components"][0]
        protein_name = cello_gate["regulator"]
        protein_sequence_names = [elem["components"][1] for elem in structure["devices"] if
                                  "cassette" in elem["name"]]

        cello_utr_part = cello_parts_by_name[utr_name]
        cello_promoter_part = [cello_parts_by_name[name] for name in promoter_sequence_names]
        cello_regulator_parts = [cello_parts_by_name[name] for name in protein_sequence_names]

        utr_sequence_entry = get_sequence_entry(name=utr_name, sequence=cello_utr_part["dnasequence"])

        protein_sequence_entries = [get_sequence_entry(name=name, sequence=cello_parts_by_name[name]["dnasequence"])
                                    for name in protein_sequence_names]
        promoter_sequence_entries = [get_sequence_entry(name=name, sequence=cello_parts_by_name[name]["dnasequence"])
                                     for name in promoter_sequence_names]

        utr_entry = {"collection": "utrs",
                     "identifier": "utr_" + utr_name,
                     "name": utr_name,
                     "sequence_id": utr_sequence_entry["identifier"]}

        cds_length = np.median([len(seq["sequence"]) for seq in protein_sequence_entries])
        rna_length = len(utr_sequence_entry["sequence"]) + cds_length
        protein_length = cds_length / 3

        protein_entry = get_protein_entry(name=protein_name,
                                          sequence_ids=[seq["identifier"] for seq in protein_sequence_entries],
                                          transcription_rate=transcription_rate,
                                          rna_degradation_rate=rna_degradation_rate,
                                          rna_length=rna_length,
                                          translation_rate=translation_rate,
                                          protein_degradation_rate=protein_degradation_rate,
                                          protein_length=protein_length)

        cognate_transcription_factors = [protein_name, protein_sequence_names]
        promoter_entry = get_promoter_entry(name=promoter_name,
                                            cognate_transcription_factors=cognate_transcription_factors,
                                            sequence_ids=[seq_name for seq_name in promoter_sequence_names],
                                            scaling_factor=eu_to_rpu_scaling_factor,
                                            num_binding_sites=num_binding_sites)
        # The input levels are in RPU. However, the gene-centric approach requires the input as either inducer levels or TF levels
        # This is achieved by using the inducer concentrations from the paper associated to the respective rpu values.
        # X = cytometry.get_input_levels()
        X = reference_inducer_inputs
        X_rpu = list(cytometry.get_input_levels())
        Y = [cytometry.get_histogram(x) for x in cytometry.get_input_levels()]
        Y = [(hist.mean(), hist.variance()) for hist in Y]
        response_characteristic = {"X": np.array(X),
                                   "X_rpu": np.array(X_rpu),
                                   "Y": np.array(Y),
                                   "labels": ["mean", "variance"]}
        input_information = {"promoter": reference_sensor_promoter_entry}
        gate_information = {  # "utr":None,
            "cds": protein_entry,
            "promoter": promoter_entry,
            "name": gate_name}

        promoter_entry = match_promoter(response_characteristic=response_characteristic,
                                        reporter_information=reporter_information,
                                        input_information=input_information,
                                        gate_information=gate_information,
                                        match_input_sensor=False)

        device_entry = get_device_entry(name=gate_name, group=cello_gate["group"],
                                        regulator=cello_gate["regulator"],
                                        primitive_identifer=["NOT", "NOR2"],
                                        utr_id=utr_entry["identifier"],
                                        cds_id=protein_entry["identifier"],
                                        promoter_id=promoter_entry["identifier"],
                                        color=cello_gate["color"])

        custom_lib += [device_entry, utr_entry, protein_entry, promoter_entry]
        # custom_lib += [device_entry, protein_entry, promoter_entry]
        custom_lib += [*promoter_sequence_entries, utr_sequence_entry, *protein_sequence_entries]

        # Next steps:
        # Match Characteristics of promoter to cytometry data
        # Infer Inflection Point
        # Store everything in a list

        # Add everything to the lists

        pass
        pass
        # Write out everything to a json file

    custom_lib = {elem["identifier"]: elem for elem in custom_lib}
    custom_lib = list(custom_lib.values())

    output_dir = "data/gate_libs/"
    with open(output_dir + "gate_lib_yeast_generated.json", "w") as file:
        json.dump(custom_lib, file, indent=4)
