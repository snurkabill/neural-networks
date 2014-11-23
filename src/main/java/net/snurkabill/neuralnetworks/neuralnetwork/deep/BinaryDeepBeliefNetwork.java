package net.snurkabill.neuralnetworks.neuralnetwork.deep;

import net.snurkabill.neuralnetworks.heuristic.HeuristicDBN;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.BinaryRestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.utilities.Utilities;
import net.snurkabill.neuralnetworks.weights.weightfactory.WeightsFactory;

public class BinaryDeepBeliefNetwork extends BinaryRestrictedBoltzmannMachine implements DeepNetwork {

    private final StuckedRBM networkChimney;
    private final int sizeOfOutputVector;
    private final double[] temporaryOutputValues;
    private double[] transformedInput;

    public BinaryDeepBeliefNetwork(String name, int numberOfOutputValues, int numOfHidden, WeightsFactory wFactory,
                                   HeuristicDBN heuristic, long seed, StuckedRBM networkChimney) {
        super(name, networkChimney.getSizeOfOutputVector() + numberOfOutputValues, numOfHidden,
                wFactory, heuristic, seed);
        this.networkChimney = networkChimney;
        this.sizeOfOutputVector = numberOfOutputValues;
        temporaryOutputValues = new double [sizeOfOutputVector];
        transformedInput = new double [networkChimney.getSizeOfOutputVector()];
    }

    private void nullOutputPartOfVector() {
        for (int i = networkChimney.getSizeOfOutputVector(), j = 0;
             i < visibleNeurons.length; i++, j++) {
            visibleNeurons[i] = 0.0;
            temporaryOutputValues[j] = 0.0;
        }
    }

    private double[] calculateNetworkChimney(double[] inputVector) {
        int numOfIterations = ((HeuristicDBN)super.getHeuristic()).numOfRunsOfNetworkChimney;
        double[] outputFromChimney = new double[networkChimney.getSizeOfOutputVector()];
        for (int i = 0; i < numOfIterations; i++) {
            networkChimney.calculateNetwork(inputVector);
            double[] tmp = networkChimney.getOutputValues();
            for (int j = 0; j < outputFromChimney.length; j++) {
                outputFromChimney[i] += tmp[i];
            }
        }
        for (int i = 0; i < outputFromChimney.length; i++) {
            outputFromChimney[i] /= numOfIterations;
        }
        return outputFromChimney;
    }

    @Override
    public void calculateNetwork(double[] inputVector) {
        transformedInput = calculateNetworkChimney(inputVector);
        int numOfIterations = ((HeuristicDBN)super.getHeuristic()).numofRunsBaseRBMItself;
        for (int i = 0; i < numOfIterations; i++) {
            System.arraycopy(transformedInput, 0, visibleNeurons, 0, networkChimney.getSizeOfOutputVector());
            nullOutputPartOfVector();
            this.machineStep();
            for (int j = 0; j < sizeOfOutputVector; j++) {
                temporaryOutputValues[j] += visibleNeurons[j + networkChimney.getSizeOfOutputVector()];
            }
        }
        for (int i = 0; i < sizeOfOutputVector; i++) {
            temporaryOutputValues[i] /= numOfIterations;
        }
    }

    @Override
    protected double calcOutputVectorError(double[] targetValues) {
        return Utilities.calcError(targetValues, temporaryOutputValues);
    }

    @Override
    public int getSizeOfOutputVector() {
        return sizeOfOutputVector;
    }

    @Override
    public int getSizeOfInputVector() {
        return networkChimney.getSizeOfInputVector();
    }

    @Override
    public double[] getOutputValues() {
        double[] outputVals = new double[sizeOfOutputVector];
        System.arraycopy(temporaryOutputValues, 0, outputVals, 0, sizeOfOutputVector);
        return outputVals;
    }

    @Override
    public void getOutputValues(double[] outputValues) {
        System.arraycopy(temporaryOutputValues, 0, outputValues, 0, sizeOfOutputVector);
    }

    @Override
    public void trainNetwork(double[] targetValues) {
        double[] fullTargetValues = new double [networkChimney.getSizeOfOutputVector() + sizeOfOutputVector];
        System.arraycopy(transformedInput, 0, fullTargetValues, 0, transformedInput.length);
        System.arraycopy(targetValues, 0, fullTargetValues, transformedInput.length, sizeOfOutputVector);
        super.trainMachine(fullTargetValues);
    }
}
