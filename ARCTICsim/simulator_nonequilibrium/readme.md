# Non-Equilibrium Steady State Simulation
The non-equilibrium steady state simulator is published in [Energy Aware Technology Mapping of Genetic Logic Circuits, Kubaczka et al. 2024](https://pubs.acs.org/doi/10.1021/acssynbio.4c00395) and was first presented at [IWBDA 2023](https://www.iwbdaconf.org/2023/).

The genetic logic circuit models are located in [simulator](ARCTICsim/simulator_nonequilibrium/simulator). The underlying models for gene expression are in the directory [models](ARCTICsim/simulator_nonequilibrium/simulator).

# Quickstart

    cd ARCTICsim/simulator_nonequilibrium/
    python3 main.py

This loads the structure and the genetic gate library defined in [settings_config.cfg](ARCTICsim/simulator_nonequilibrium/settings_config.cfg).
Thereafter, you can start a simulation with the default arguments by using the command `start` and providing an assignment via the `-a` argument.

    start -a={"a":"input_3","b":"input_1","c":"input_2","NOT_0":"P1_PhlF","NOT_2":"P1_IcaR","NOT_4":"P1_PsrA","NOR2_1":"P1_QacR","NOR2_3":"P1_HKCI","O":"output_1"}

Use the deterministic value propagation via command line

    start -a={"a":"input_3","b":"input_1","c":"input_2","NOT_0":"P1_PhlF","NOT_2":"P1_IcaR","NOT_4":"P1_PsrA","NOR2_1":"P1_QacR","NOR2_3":"P1_HKCI","O":"output_1"} -m=det

Use the sampling based value propagation via command line with `1000` samples

    start -a={"a":"input_3","b":"input_1","c":"input_2","NOT_0":"P1_PhlF","NOT_2":"P1_IcaR","NOT_4":"P1_PsrA","NOR2_1":"P1_QacR","NOR2_3":"P1_HKCI","O":"output_1"} -m=samp -n=1000

and with `100` samples

    start -a={"a":"input_3","b":"input_1","c":"input_2","NOT_0":"P1_PhlF","NOT_2":"P1_IcaR","NOT_4":"P1_PsrA","NOR2_1":"P1_QacR","NOR2_3":"P1_HKCI","O":"output_1"} -m=samp -n=100

# Settings
Within [settings_config.cfg](ARCTICsim/simulator_nonequilibrium/settings_config.cfg), you find the default settings of the simulator and are able to adapt them to your needs.
These can also be provided at startup time via `-x`with `x` being the first letter of the property to change.
This is especially relevant to provide the structure to use.

In combination with the `start` command, only the properties `assignment` (`-a`), `num_samples` (`-n`), and `mode` (`-m`) are of relevance. For subsequent tasks, one can make use of the option `output` (`-o`). 


| Property    | Type                          | Description                                                                                                                                              | Example         |
|-------------|-------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------|
| assignment  | json dict without whitespaces | The assignment to simulate.                                                                                                                              | See quickstart. |
| mode        | string                        | The mode of operation. `det` for deterministic simulation based on the expected value. `samp` is for sampling based function simulation.                 | `-m=samp`       |
| num_samples | positive int                  | The number of samples to use for estimating the circuit output distribution in the case of `mode = samp`.                                                | `-n=1000`       |
| output      | string                        | Wether to return only score values from the simulation (`score`) or the score and all simulation values and further information in the circuit (`full`). | `-o=score`      |