package net.snurkabill.neuralnetworks.weights.weightfactory;

import net.snurkabill.neuralnetworks.neuralnetwork.deep.DeepBoltzmannMachine;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.FeedForwardNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction.TransferFunctionCalculator;

public class FineTuningWeightsFactory implements WeightsFactory {

    private final double[][][] weights;

    public FineTuningWeightsFactory(DeepBoltzmannMachine deepBoltzmannMachine, FeedForwardNetwork backNetwork) {
        if (deepBoltzmannMachine.getForwardedValues().length != backNetwork.getSizeOfInputVector()) {
            throw new IllegalArgumentException("No compatibility!");
        }
        double[][][] dbmWeights = deepBoltzmannMachine.getWeights();
        double[][][] backWeights = backNetwork.getWeights();
        int numOfLayers = dbmWeights.length + backWeights.length;
        this.weights = new double[numOfLayers][][]; // +1 for linearClassifier layer
        for (int i = 0; i < dbmWeights.length; i++) {
            this.weights[i] = new double[dbmWeights[i].length][];
        }
        for (int i = dbmWeights.length, j = 0; i < numOfLayers; i++, j++) {
            this.weights[i] = new double[backWeights[j].length][];
        }
        for (int i = 0; i < dbmWeights.length; i++) {
            System.arraycopy(dbmWeights[i], 0, this.weights[i], 0, this.weights[i].length);
        }
        for (int i = dbmWeights.length, j = 0; i < numOfLayers; i++, j++) {
            this.weights[i] = backWeights[j];
        }
    }

    public FineTuningWeightsFactory(DeepBoltzmannMachine deepBoltzmannMachine, int numOfClassesInLinearClassifier,
                                    long seed, TransferFunctionCalculator transferFunctionCalculator) {
        double[][][] dbmWeights = deepBoltzmannMachine.getWeights();
        int numOfLayers = dbmWeights.length + 1;
        this.weights = new double[numOfLayers][][]; // +1 for linearClassifier layer
        for (int i = 0; i < dbmWeights.length; i++) {
            this.weights[i] = new double[dbmWeights[i].length][];
        }
        this.weights[dbmWeights.length] = new double[dbmWeights[dbmWeights.length - 1][0].length + 1][];
        for (int i = 0; i < dbmWeights.length; i++) {
            System.arraycopy(dbmWeights[i], 0, this.weights[i], 0, this.weights[i].length);
        }
        for (int i = 0; i < weights[weights.length - 1].length; i++) {
            weights[weights.length - 1][i] = new double[numOfClassesInLinearClassifier];
        }

        WeightsFactory factory = new SmartGaussianRndWeightsFactory(transferFunctionCalculator, seed);
        for (int i = 0; i < weights[weights.length - 1].length; i++) {
            factory.setWeights(this.weights[numOfLayers - 1][i]);
        }
    }

    public FineTuningWeightsFactory(DeepBoltzmannMachine deepBoltzmannMachine, int numOfClassesInLinearClassifier,
                                    long seed, double weightsScale) {
        double[][][] dbmWeights = deepBoltzmannMachine.getWeights();
        int numOfLayers = dbmWeights.length + 1;
        this.weights = new double[numOfLayers][][]; // +1 for linearClassifier layer
        for (int i = 0; i < dbmWeights.length; i++) {
            this.weights[i] = new double[dbmWeights[i].length][];
        }
        this.weights[dbmWeights.length] = new double[dbmWeights[dbmWeights.length - 1][0].length + 1][];
        for (int i = 0; i < dbmWeights.length; i++) {
            System.arraycopy(dbmWeights[i], 0, this.weights[i], 0, this.weights[i].length);
        }
        for (int i = 0; i < weights[weights.length - 1].length; i++) {
            weights[weights.length - 1][i] = new double[numOfClassesInLinearClassifier];
        }

        WeightsFactory factory = new GaussianRndWeightsFactory(weightsScale, seed);
        for (int i = 0; i < weights[weights.length - 1].length; i++) {
            factory.setWeights(this.weights[numOfLayers - 1][i]);
        }
    }

    @Override
    public void setWeights(double[][][] weights) {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                System.arraycopy(this.weights[i][j], 0, weights[i][j], 0, weights[i][j].length);
            }
        }
    }

    @Override
    public void setWeights(double[][] weights) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void setWeights(double[] weights) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
