package net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel;

import net.snurkabill.neuralnetworks.neuralnetwork.NeuralNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.EnergyBased;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction.SigmoidFunction;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction.TransferFunctionCalculator;

import java.util.Random;

public abstract class BoltzmannMachine extends NeuralNetwork implements EnergyBased {

    private final TransferFunctionCalculator probabilityCalculator = new SigmoidFunction();
    protected double temperature = 1;
    protected final Random random;

    public BoltzmannMachine(String name, long seed) {
        super(name);
        this.random = new Random(seed);
    }

    protected double calcProbabilityOfPositiveOutput(double potential) {
        return probabilityCalculator.calculateOutputValue(potential / temperature);
    }

    public double getTemperature() {
        return temperature;
    }

    public void setTemperature(double temperature) {
        LOGGER.debug("Setting temperature to: {}", temperature);
        this.temperature = temperature;
    }

    public TransferFunctionCalculator getProbabilityCalculator() {
        return probabilityCalculator;
    }
}
