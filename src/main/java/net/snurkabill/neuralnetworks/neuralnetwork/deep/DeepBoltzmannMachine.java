package net.snurkabill.neuralnetworks.neuralnetwork.deep;

import net.snurkabill.neuralnetworks.neuralnetwork.FeedForwardable;
import net.snurkabill.neuralnetworks.neuralnetwork.NeuralNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction.TransferFunctionCalculator;
import net.snurkabill.neuralnetworks.utilities.DeepNetUtils;
import net.snurkabill.neuralnetworks.utilities.Utilities;

import java.util.List;
import net.snurkabill.neuralnetworks.heuristic.Heuristic;

public class DeepBoltzmannMachine extends NeuralNetwork implements DeepNetwork, FeedForwardable {

    private final List<RestrictedBoltzmannMachine> machines;
    private final int indexOfLastLayer;
    private final int numOfLayers;
    private final int sizeOfOutputVector;
    private final int sizeOfInputVector;

    public DeepBoltzmannMachine(String name, List<RestrictedBoltzmannMachine> machines) {
        super(name);
        DeepNetUtils.checkMachineList(machines);
        this.machines = machines;
        this.numOfLayers = machines.size();
        this.indexOfLastLayer = machines.size() - 1;
        this.sizeOfInputVector = machines.get(0).getSizeOfInputVector();
        this.sizeOfOutputVector = machines.get(indexOfLastLayer).getSizeOfOutputVector();
    }

    public double[][][] getWeights() {
        double[][][] weights = new double[machines.size()][][];
        for (int i = 0; i < machines.size(); i++) {
            for (int j = 0; j < machines.get(i).getSizeOfVisibleVector(); j++) {
                weights[i] = machines.get(i).getWeights();
            }
        }
        return weights;
    }

    @Override
    public void feedForward(double[] inputVector) {
        machines.get(0).setVisibleNeurons(inputVector);
        for (int i = 0; i < numOfLayers - 1; i++) {
            machines.get(i + 1).setVisibleNeurons(machines.get(i).activateHiddenNeurons());
        }
        machines.get(indexOfLastLayer).activateHiddenNeurons();
    }

    @Override
    public double[] getForwardedValues() {
        return machines.get(indexOfLastLayer).getHiddenNeurons();
    }

    @Override
    public void calculateNetwork(double[] inputVector) {
        feedForward(inputVector);
        for (int i = indexOfLastLayer; i > 0; i--) {
            machines.get(i - 1).setHiddenNeurons(machines.get(i).activateVisibleNeurons());
        }
        machines.get(0).activateVisibleNeurons();
    }

    @Override
    protected double calcOutputVectorError(double[] targetValues) {
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
    public TransferFunctionCalculator getTransferFunction() {
        return machines.get(0).getTransferFunction();
    }

	@Override
	public void trainNetwork(double[] targetValues) {
		throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
	}

	@Override
	public Heuristic getHeuristic() {
		throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
	}

	@Override
	public void setHeuristic(Heuristic heuristic) {
		throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
	}

}
