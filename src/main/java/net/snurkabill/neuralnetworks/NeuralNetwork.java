package net.snurkabill.neuralnetworks;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public abstract class NeuralNetwork {

    private final String name;
    private final String workName;
    protected final Logger LOGGER;

    public NeuralNetwork(String name, String workingString) {
        this.name = name;
        this.workName = workingString + " " + name;
        this.LOGGER = LoggerFactory.getLogger(workName);
    }

    public String getName() {
        return name;
    }

    public String getWorkName() {
        return workName;
    }
}
