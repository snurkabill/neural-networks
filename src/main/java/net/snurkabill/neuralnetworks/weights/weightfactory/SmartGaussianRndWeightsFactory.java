package net.snurkabill.neuralnetworks.weights.weightfactory;

import net.snurkabill.neuralnetworks.math.function.transferfunction.TransferFunctionCalculator;

import java.util.Random;

public class SmartGaussianRndWeightsFactory implements WeightsFactory {

    // TODO: RENAME ME! :D
    private static final double KIND_OF_MAGIC_NUMBER = 3.0;

    private final TransferFunctionCalculator transferFunctionCalculator;
    private final Random random;

    public SmartGaussianRndWeightsFactory(TransferFunctionCalculator transferFunctionCalculator,
                                          long seed) {
        this.transferFunctionCalculator = transferFunctionCalculator;
        this.random = new Random(seed);
    }

    @Override
    public void setWeights(double[][][] weights) {
        for (double[][] weight : weights) {
            setWeights(weight);
        }
    }

    @Override
    public void setWeights(double[][] weights) {
        for (double[] weight : weights) {
            setWeights(weight);
        }
    }

    @Override
    public void setWeights(double[] weights) {
        double bound = Math.sqrt(KIND_OF_MAGIC_NUMBER / weights.length) /
                transferFunctionCalculator.getPositiveArgumentRange();
        for (int k = 0; k < weights.length; k++) {
            double newWeight = random.nextDouble() * bound;
            if (random.nextInt(2) == 0) {
                weights[k] = -newWeight;
            } else {
                weights[k] = newWeight;
            }
        }
    }
}
