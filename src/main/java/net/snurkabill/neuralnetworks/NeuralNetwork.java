package net.snurkabill.neuralnetworks;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class NeuralNetwork {

    private final String name;
    private final String fullName;
    protected final Logger LOGGER;

    public NeuralNetwork(String name, String typeOfNeuralNetwork) {
        this.name = name;
        this.fullName = typeOfNeuralNetwork + " " + name;
        this.LOGGER = LoggerFactory.getLogger(fullName);
    }

    public String getName() {
        return name;
    }

    public String getFullName() {
        return fullName;
    }
}
