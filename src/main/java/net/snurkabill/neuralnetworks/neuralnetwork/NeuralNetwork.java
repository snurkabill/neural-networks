package net.snurkabill.neuralnetworks.neuralnetwork;

import net.snurkabill.neuralnetworks.heuristic.BasicHeuristic;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction.TransferFunctionCalculator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class NeuralNetwork {

    private final String name;
    protected final String workingName;
    protected final Logger LOGGER;

    public NeuralNetwork(String name) {
        this.name = name;
        this.workingName = determineWorkingName();
        this.LOGGER = LoggerFactory.getLogger(name);
    }

    public String getName() {
        return name;
    }

    public String getWorkingName() {
        return workingName;
    }

    public double calcError(double[] inputVector, double[] targetValues) {
        calculateNetwork(inputVector);
        return calcOutputVectorError(targetValues);
    }

    public abstract void calculateNetwork(double[] inputVector);

    // TODO: implement method right here
    protected abstract double calcOutputVectorError(double[] targetValues);

    public abstract int getSizeOfOutputVector();

    public abstract int getSizeOfInputVector();

    public abstract double[] getOutputValues();

    public abstract void getOutputValues(double[] outputValues);

    public abstract TransferFunctionCalculator getTransferFunction();

    public abstract void trainNetwork(double[] targetValues);

    public abstract BasicHeuristic getHeuristic();

    public abstract void setHeuristic(BasicHeuristic heuristic);

    public abstract double[][][] getWeights();

    public String determineWorkingName() {
        return "Name: " + this.name;
    }
}
