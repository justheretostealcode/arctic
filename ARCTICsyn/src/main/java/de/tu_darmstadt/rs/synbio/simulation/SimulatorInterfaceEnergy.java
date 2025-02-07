package de.tu_darmstadt.rs.synbio.simulation;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import de.tu_darmstadt.rs.synbio.common.circuit.Circuit;
import de.tu_darmstadt.rs.synbio.common.library.GateLibrary;
import de.tu_darmstadt.rs.synbio.mapping.Assignment;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class SimulatorInterfaceEnergy {

    private static final Logger logger = LoggerFactory.getLogger(SimulatorInterfaceEnergy.class);

    private static final String scorePrefix = ":> {\"functional_score\":";

    private final String pythonBinary;
    private final File simulatorPath;
    private final String simScript;
    private final String simInitArgs;
    private final String simArgs;
    private final GateLibrary library;

    private ProcessBuilder pb;
    private Process simProcess;
    private File structureFile;
    private BufferedReader reader;
    private BufferedWriter writer;
    private BufferedReader errorReader;
    private final ObjectMapper mapper = new ObjectMapper();
    private double energy;

    private Circuit currentCicuit;

    public SimulatorInterfaceEnergy(SimulationConfiguration config, GateLibrary gateLibrary) {
        pythonBinary = config.getPythonBinary();
        simulatorPath = config.getSimPath();
        simScript = config.getSimScript();
        simInitArgs = config.getSimInitArgs();
        simArgs = config.getSimArgs();
        library = gateLibrary;
    }

    public boolean initSimulation(Circuit circuit) { //TODO: handle return value

        currentCicuit = circuit;

        if (simProcess!= null && simProcess.isAlive())
            simProcess.destroy();

        try {
            String structureFileName = "structure_" + circuit.getIdentifier() + "_tid" + Thread.currentThread().getId() + "_" + System.nanoTime() + ".json";
            structureFile = new File(simulatorPath, structureFileName);
            circuit.save(structureFile);

            List<String> arguments = new ArrayList<>();
            arguments.addAll(Arrays.asList(pythonBinary, simScript, "-s=" + structureFileName, "-l=" + library.getSourceFile().getAbsolutePath()));
            arguments.addAll(Arrays.asList(simInitArgs.split(" ")));

            pb = new ProcessBuilder(arguments.toArray(new String[0]));
            pb.directory(simulatorPath);

            simProcess = pb.start();

            reader = new BufferedReader(new InputStreamReader(simProcess.getInputStream()));
            writer = new BufferedWriter(new OutputStreamWriter(simProcess.getOutputStream()));
            errorReader = new BufferedReader(new InputStreamReader(simProcess.getErrorStream()));

            String output;
            do {
                output = reader.readLine();

                if (!simProcess.isAlive()) {
                    logger.error("Simulator exited on initialization:\n" + getError());
                    shutdown();
                    return false;
                }

            } while (output == null || !output.startsWith("ready"));

            return true;

        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    public Double simulate(Assignment assignment) {

        double score = -1.0;

        try {

            if (!simProcess.isAlive()) {
                logger.error("Simulator exited before simulation start:\n" + getError());
                shutdown();
                return -1.19291923;
            }
            // long start = System.nanoTime();

            String assignmentStr = mapper.writeValueAsString(assignment.getIdentifierMap());

            writer.write("start " + simArgs + " -a=" + assignmentStr);
            writer.newLine();
            writer.flush();

            String output;
            do {
                output = reader.readLine();

                if (!simProcess.isAlive()) {
                    logger.error("Simulator exited during simulation: " + getError());
                    logger.error("Assignment:"  + assignmentStr);
                    logger.error("Structure: " + currentCicuit.getIdentifier());
                    shutdown();
                    return 0.0;
                }
            } while (output == null || !output.startsWith(scorePrefix));

            output = output.substring(2);

            JsonNode result;

            try {
                result = mapper.readTree(output);
            } catch (Exception e) {
                //e.printStackTrace();
                //logger.error("Assignment:"  + assignmentStr);
                //logger.error("Structure: " + currentCicuit.getIdentifier());
                return -1.0;
            }

            score = result.get("functional_score").iterator().next().asDouble();
            setEnergy(result.get("energy_score").asDouble());

            // long end = System.nanoTime();
            // long duration = end - start;
            // System.out.printf("Duration: %f\n", duration * Math.pow(10, -9));

        } catch (Exception e) {
            e.printStackTrace();
            shutdown();
        }

        return score;
    }

    public Double simulateWithRetry(Assignment assignment) {

        double score = simulate(assignment);

        if (score == -1.0) {
            //logger.error("Retrying simulation.");
            score = simulate(assignment);
        }

        if (score == -1.0)
            return 0.0;
        else
            return score;
    }

    private void setEnergy(double energy) {
        this.energy = energy;
    }

    public double getEnergy() {
        return energy;
    }

    public void shutdown() {
        structureFile.delete();
        simProcess.destroy();
    }

    private String getError() throws IOException {
        String line;
        StringBuilder error = new StringBuilder();
        while (errorReader.ready() && (line = errorReader.readLine()) != null) {
            error.append(line).append("\n");
        }
        return error.toString();
    }
}
