package net.snurkabill.neuralnetworks.weights.weightfactory;

import net.snurkabill.neuralnetworks.neuralnetwork.deep.DeepBoltzmannMachine;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction.TransferFunctionCalculator;

public class FineTuningWeightsFactory implements WeightsFactory {

    private final double[][][] weights;

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
        for (int i = 0; i < numOfClassesInLinearClassifier; i++) {
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
