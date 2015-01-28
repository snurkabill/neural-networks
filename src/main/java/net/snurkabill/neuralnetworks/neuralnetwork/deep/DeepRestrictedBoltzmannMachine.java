package net.snurkabill.neuralnetworks.neuralnetwork.deep;

import net.snurkabill.neuralnetworks.heuristic.Heuristic;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.BoltzmannMachine;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.utilities.Utilities;

import java.util.List;

public class DeepRestrictedBoltzmannMachine extends BoltzmannMachine implements DeepNetwork {

    private final List<RestrictedBoltzmannMachine> machines;
    private final List<Integer> iterationsForTraining;
    private int trainedLayer = 0;
    private final int numOfLayers;
    private final int indexOfLastLayer;
    private final int sizeOfInputVector;
    private final int sizeOfOutputVector;

    public DeepRestrictedBoltzmannMachine(String name, long seed, List<RestrictedBoltzmannMachine> machines,
                                          List<Integer> iterationsForTraining) {
        super(name, seed);
        this.machines = machines;
        this.numOfLayers = machines.size();
        this.indexOfLastLayer = machines.size() - 1;
        this.sizeOfInputVector = machines.get(0).getSizeOfInputVector();
        this.sizeOfOutputVector = machines.get(indexOfLastLayer).getSizeOfOutputVector();
        this.iterationsForTraining = iterationsForTraining;
        if(iterationsForTraining.size() != machines.size()) {
            throw new IllegalArgumentException("Size of list of iterations is different form list of machines");
        }
    }

    @Override
    public double calcEnergy(double[] inputVector) {
        return 0;
    }

    @Override
    public void calculateNetwork(double[] inputVector) {
        machines.get(0).setVisibleNeurons(inputVector);
        for (int i = 0; i < numOfLayers - 1; i++) {
            machines.get(i + 1).setVisibleNeurons(machines.get(i).activateHiddenNeurons());
        }
        machines.get(indexOfLastLayer).activateHiddenNeurons();
        for (int i = indexOfLastLayer; i > 0; i--) {
            machines.get(i - 1).setHiddenNeurons(machines.get(i).activateVisibleNeurons());
        }
        machines.get(0).activateVisibleNeurons();
    }

    @Override
    protected double calcOutputVectorError(double[] targetValues) {
        if (targetValues.length != sizeOfInputVector) {
            throw new IllegalArgumentException();
        }
        return Utilities.calcError(targetValues, machines.get(0).getVisibleNeurons());
    }

    @Override
    public int getSizeOfOutputVector() {
        return sizeOfInputVector;
    }

    @Override
    public int getSizeOfInputVector() {
        return sizeOfInputVector;
    }

    public double[] getLastHiddenValues() {
        return machines.get(indexOfLastLayer).getHiddenNeurons();
    }

    public void getLastHiddenValues(double[] outputValues) {
        if(outputValues.length != machines.get(indexOfLastLayer).getSizeOfHiddenVector()) {
            throw new IllegalArgumentException("Wrong dimension of vector");
        }
        double[] output = machines.get(0).getHiddenNeurons();
        System.arraycopy(output, 0, outputValues, 0, outputValues.length);
    }

    @Override
    public double[] getOutputValues() {
        return machines.get(0).getVisibleNeurons();
    }

    @Override
    public void getOutputValues(double[] outputValues) {
        if(outputValues.length != this.sizeOfInputVector) {
            throw new IllegalArgumentException("Wrong dimension of vector");
        }
        double[] output = machines.get(0).getVisibleNeurons();
        System.arraycopy(output, 0, outputValues, 0, outputValues.length);
    }

    @Override
    public void trainNetwork(double[] targetValues) {
        machines.get(0).setVisibleNeurons(targetValues);
        if(iterationsForTraining.get(trainedLayer) == 0 && trainedLayer != indexOfLastLayer) {
            trainedLayer++;
        }
        for (int i = 0; i < trainedLayer; i++) {
            machines.get(i + 1).setVisibleNeurons(machines.get(i).activateHiddenNeurons());
        }
        machines.get(trainedLayer).trainMachine();
    }

    @Override
    public Heuristic getHeuristic() {
        return null;
    }

    @Override
    public void setHeuristic(Heuristic heuristic) {
    }
}
