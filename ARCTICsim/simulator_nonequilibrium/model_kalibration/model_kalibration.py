from pathlib import Path

def get_default_Simulator_Settings(name: str) -> None:
    pass

def get_Trainging_Data(path: Path) -> None:
    pass

def get_Simulator(name: str) -> None:
    pass

def model_kalibration(simulator, trainings_data, simulator_settings) -> None:
    pass


#args
simulator_name = "Default"
trainings_data_path: Path = "ARCTICsim/simulator_nonequilibrium/data/yeast/SC1C1G1T1.UCF.json"
simulator_settings = {}

if __name__ == "__main__":

    trainings_data = get_Trainging_Data(trainings_data_path)

    simulator = get_Simulator(simulator_name)

    if simulator_settings == None:
        simulator_settings = get_default_Simulator_Settings(simulator_name)

    result = model_kalibration(simulator, trainings_data, simulator_settings)

    print(result)