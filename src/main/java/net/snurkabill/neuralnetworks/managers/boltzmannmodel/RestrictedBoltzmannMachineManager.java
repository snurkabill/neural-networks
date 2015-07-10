package net.snurkabill.neuralnetworks.managers.boltzmannmodel;

import net.snurkabill.neuralnetworks.data.database.ClassFullDatabase;
import net.snurkabill.neuralnetworks.heuristic.calculators.HeuristicCalculator;
import net.snurkabill.neuralnetworks.managers.NetworkManager;
import net.snurkabill.neuralnetworks.neuralnetwork.NeuralNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;

import java.util.Random;

public abstract class RestrictedBoltzmannMachineManager extends NetworkManager {

    protected final Random random;

    public RestrictedBoltzmannMachineManager(NeuralNetwork neuralNetwork, ClassFullDatabase classFullDatabase, long seed,
                                             HeuristicCalculator heuristicCalculator) {
        super(neuralNetwork, classFullDatabase, heuristicCalculator);
        this.random = new Random(seed);
    }

    public double[] sampleHidden(double[] inputVector, int iterations) {
        RestrictedBoltzmannMachine machine = (RestrictedBoltzmannMachine)neuralNetwork;
        double[] outputVector = new double[machine.getSizeOfHiddenVector()];
        for (int i = 0; i < iterations; i++) {
            machine.feedForward(inputVector);
            double[] outputVals = machine.getForwardedValues();
            for (int j = 0; j < outputVector.length; j++) {
                outputVector[i] += outputVals[i];
            }
        }
        for (int i = 0; i < outputVector.length; i++) {
            outputVector[i] /= iterations;
        }
        return outputVector;
    }

}
