import json
import os

import numpy as np

import parasbolv as psv
import matplotlib.pyplot as plt
from collections import namedtuple
from math import cos, sin, pi, sqrt

"""
Visualization of Genetic Logic Circuits consisting of two plasmids with SBOL Visual on the basis of paraSBOLv.

Reference paraSBOLv
Clark C.J., Scott-Brown J. & Gorochowski T.E. "paraSBOLv: a foundation for standard-compliant genetic design visualisation tools", Synthetic Biology, 2021 doi:10.1093/synbio/ysab022

paraSBOLv has been slightly adapted to allow for the needs of multi plasmid visualizations. In particular, it returns the bound_list of the glyphs included in a construct besides the bounds of the construct itself.
"""

Part = namedtuple('part', ['glyph_type', 'orientation', 'user_parameters', 'style_parameters'])
Interaction = namedtuple('interaction',
                         ['starting_glyph', 'ending_glyph', 'interaction_type', 'interaction_parameters'])


def draw_interaction(ax,
                     sending_bounds,
                     receiving_bounds,
                     interaction_type,
                     parameters,
                     sending_baseline=None,
                     receiving_baseline=None,
                     connector_level=None,
                     rotation=0.0):
    """
    This code is an adaption of the code provided in parasbolv.py to support multi plasmid interactions.
    The respective license accompanying paraSBOLv is

    ##################################################################################################
    MIT License

    Copyright (c) 2021 Biocompute Lab

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    ##################################################################################################

    :param ax:
    :param sending_bounds:
    :param receiving_bounds:
    :param interaction_type:
    :param parameters:
    :param rotation:
    :return:
    """
    parameters = psv.process_interaction_params(parameters)
    # Convert to degrees
    rotation = (180 / pi) * rotation

    # Determine distance from baseline
    initial_distance = parameters['distance_from_baseline']
    if interaction_type == 'degradation':  # Degradation is bigger than the other interactions
        initial_distance = parameters['distance_from_baseline'] * 2
    # Determine pad size
    y_pad = parameters['heightskew'] * 2
    # Find centroid of glyph bounds
    origin_cent = (sending_bounds[0][0] + (sending_bounds[1][0] - sending_bounds[0][0]) / 2,
                   sending_bounds[0][1] + (sending_bounds[1][1] - sending_bounds[0][1]) / 2)
    target_cent = (receiving_bounds[0][0] + (receiving_bounds[1][0] - receiving_bounds[0][0]) / 2,
                   receiving_bounds[0][1] + (receiving_bounds[1][1] - receiving_bounds[0][1]) / 2)

    if target_cent[1] == origin_cent[1]:
        psv.draw_interaction(ax,
                             sending_bounds,
                             receiving_bounds,
                             interaction_type,
                             parameters,
                             rotation=0.0)

    target_is_below = target_cent[1] < origin_cent[1]

    rotation = rotation % 360
    bearing = 360 - rotation

    if sending_baseline is None:
        sending_baseline = origin_cent[1]

    if receiving_baseline is None:
        sending_baseline = target_cent[1]

    if connector_level is None:
        connector_level = 0.5 * (sending_baseline + receiving_baseline)

    factor_origin = -1 if target_is_below else 1
    factor_target = 1 if target_is_below else -1

    int_origin_x = (origin_cent[0]
                    + factor_origin * (initial_distance + parameters['distance_from_baseline']) * sin(
                bearing * pi / 180))
    int_origin_y = (sending_baseline
                    + factor_origin * (initial_distance + parameters['distance_from_baseline']) * cos(
                bearing * pi / 180))

    int_end_x = (target_cent[0]
                 + factor_target * (initial_distance + parameters['distance_from_baseline']) * sin(bearing * pi / 180))
    int_end_y = (receiving_baseline
                 + factor_target * (initial_distance + parameters['distance_from_baseline']) * cos(bearing * pi / 180))

    # Determine origin max
    int_origin_max = (
        int_origin_x + factor_origin * (y_pad + parameters['sending_length_skew']) * sin(bearing * pi / 180),
        connector_level)
    # Determine end max
    int_end_max = (int_end_x + factor_target * (y_pad + parameters['sending_length_skew']) * sin(bearing * pi / 180),
                   connector_level)

    # Determine interaction endpoint

    plt.plot([int_origin_x,
              int_origin_max[0],
              int_end_max[0],
              int_end_x],
             [int_origin_y,
              int_origin_max[1],
              int_end_max[1],
              int_end_y],
             color=parameters['color'],
             lw=parameters['linewidth'],
             zorder=parameters['zorder'] - 5)  # Slightly lower zorder than head to prevent overlap

    psv.draw_inhibition(ax=ax,
                        int_end_x=int_end_x,
                        int_end_y=int_end_y,
                        parameters=parameters)


def visualize_plasmids(plasmids, name, output_dir):
    def parse_gene(gene):
        gene_dict = {elem.split("=")[0]: elem.split("=")[1] for elem in plasmids[0][2].split(" ") if "=" in elem}
        return gene_dict

    plasmid_offset = 80
    n_plasmids = len(plasmids)
    baselines = [- plasmid_offset * iX for iX in range(n_plasmids)]

    part_lists = []
    for plasmid in plasmids:
        part_list = []
        for gene in plasmid:
            if gene is None:
                continue
            gene_dict = parse_gene(gene)
            part_list.append(Part("Promoter", "forward", None, None))
            part_list.append(Part("RibosomeEntrySite", "forward", None, None))
            part_list.append(Part('CDS',
                                  'forward',
                                  None,
                                  {'cds': {'facecolor': (1, 0.5, 0.5), 'edgecolor': (1, 0, 0), 'linewidth': 2}}
                                  ))

            part_list.append(Part('Terminator', 'forward', None, None))
        part_lists.append(part_list)
    renderer = psv.GlyphRenderer()
    x_lims = []
    y_lims = []
    fig, ax = plt.subplots()
    for iX in range(n_plasmids):
        start_position = (0, baselines[iX])
        construct = psv.Construct(part_list, renderer, interaction_list=None, fig=fig, ax=ax,
                                  start_position=start_position)
        fig, ax, baseline_start, baseline_end, bounds1, bounds_list1 = construct.draw(draw_for_bounds=False)
        ax.plot([baseline_start[0], baseline_end[0]], [baseline_start[1], baseline_end[1]], color=(0, 0, 0), linewidth=1.5,
                zorder=0)

        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        x_lims.append(x_lim)
        y_lims.append(y_lim)



    ax.set_xlim((np.min(x_lims), np.max(x_lims)))
    ax.set_ylim((np.min(y_lims), np.max(y_lims)))
    plt.show()
    pass


def multi_plasmid():
    baseline_1 = 0
    baseline_2 = -80

    part_list = []
    part_list.append(Part("Promoter", "forward", None, None))
    part_list.append(Part('RibosomeEntrySite', 'forward', None, None))
    part_list.append(Part('CDS',
                          'forward',
                          None,
                          {'cds': {'facecolor': (1, 0.5, 0.5), 'edgecolor': (1, 0, 0), 'linewidth': 2}}
                          ))

    part_list.append(Part('Terminator', 'forward', None, None))
    part_list.append(Part("Promoter", "forward", None, None))

    part_list.append(Part('RibosomeEntrySite', 'forward', None, None))
    part_list.append(Part('CDS',
                          'forward',
                          None,
                          {'cds': {'facecolor': (0.5, 0.5, 1), 'edgecolor': (0, 0, 1), 'linewidth': 2}}
                          ))

    part_list.append(Part('Terminator', 'forward', None, None))

    # Create renderer
    renderer = psv.GlyphRenderer()

    # Create list of interactions to pass to render_part_list
    interaction_list = []
    interaction_list.append(Interaction(part_list[2], part_list[4], 'inhibition', {'color': (0.75, 0, 0)}))
    # Multi Plasmid Example

    fig, axes = plt.subplots(ncols=1, nrows=1)
    ax = axes

    start_position = (0, baseline_1)
    construct = psv.Construct(part_list, renderer, interaction_list=interaction_list, fig=fig, ax=ax,
                              start_position=start_position)
    fig, ax, baseline_start, baseline_end, bounds1, bounds_list1 = construct.draw(draw_for_bounds=False)
    ax.plot([baseline_start[0], baseline_end[0]], [baseline_start[1], baseline_end[1]], color=(0, 0, 0), linewidth=1.5,
            zorder=0)

    x_lim1 = ax.get_xlim()
    y_lim1 = ax.get_ylim()

    start_position = (0, baseline_2)
    construct = psv.Construct(part_list, renderer, interaction_list=interaction_list, fig=fig, ax=ax,
                              start_position=start_position)
    fig, ax, baseline_start, baseline_end, bounds2, bounds_list2 = construct.draw(draw_for_bounds=False)
    ax.plot([baseline_start[0], baseline_end[0]], [baseline_start[1], baseline_end[1]], color=(0, 0, 0), linewidth=1.5,
            zorder=0)
    draw_interaction(ax=ax,
                     sending_bounds=bounds_list1[2],
                     receiving_bounds=bounds_list2[0],
                     sending_baseline=baseline_1,
                     receiving_baseline=baseline_2,
                     connector_level=np.mean([baseline_1, baseline_2]),
                     interaction_type="inhibition",
                     parameters={"color": (0, 0, 0)},
                     )
    psv.draw_interaction(ax=ax,
                         sending_bounds=bounds_list1[2],
                         receiving_bounds=bounds_list2[0],
                         interaction_type="inhibition",
                         parameters={"color": (0, 0, 0)},
                         )

    x_lim2 = ax.get_xlim()
    y_lim2 = ax.get_ylim()

    x_lims = [x_lim1, x_lim2]
    y_lims = [y_lim1, y_lim2]

    x_start = np.min(x_lims)
    x_end = np.max(x_lims)
    y_start = np.min(y_lims)
    y_end = np.max(y_lims)

    ax.set_xlim([x_start, x_end])
    ax.set_ylim([y_start, y_end])

    # plt.savefig("multi_plasmid_example.pdf", dpi=300)

    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plasmid_directory = "data/11111101/plasmids/"
    output_dir = plasmid_directory

    # multi_plasmid()
    # exit(0)

    for file in os.listdir(plasmid_directory):
        file_path = plasmid_directory + file
        name, extension = os.path.splitext(file)

        if extension != ".json":
            continue

        print(file, extension)

        with open(file_path, "r") as file:
            plasmids = json.load(file)
        visualize_plasmids(plasmids, name, output_dir)
        break
    pass
