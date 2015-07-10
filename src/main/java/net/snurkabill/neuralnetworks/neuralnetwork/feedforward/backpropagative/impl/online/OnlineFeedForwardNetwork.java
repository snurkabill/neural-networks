package net.snurkabill.neuralnetworks.neuralnetwork.feedforward.backpropagative.impl.online;

import net.snurkabill.neuralnetworks.heuristic.FeedForwardHeuristic;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.backpropagative.BackPropagative;
import net.snurkabill.neuralnetworks.math.function.transferfunction.TransferFunctionCalculator;
import net.snurkabill.neuralnetworks.weights.weightfactory.WeightsFactory;

import java.util.List;

public class OnlineFeedForwardNetwork extends BackPropagative {

    public OnlineFeedForwardNetwork(String name, List<Integer> topology, WeightsFactory wFactory,
                                    FeedForwardHeuristic heuristic, TransferFunctionCalculator transferFunction) {
        super(name, topology, wFactory, heuristic, transferFunction);
    }

    @Override
    protected void calcGradients(double[] targetValues) {
        for (int i = 0; i < topology[lastLayerIndex]; i++) {
            gradients[lastLayerIndex][i] = (targetValues[i] - neuronOutputValues[lastLayerIndex][i]) *
                    transferFunction.calculateDerivative(neuronOutputValues[lastLayerIndex][i]);
        }
        for (int i = numOfLayers - 2; i > 0; i--) {
            for (int j = 0; j < topology[i]; j++) {
                double sum = 0.0;
                for (int k = 0; k < topology[i + 1]; k++) {
                    sum += weights[i][j][k] * gradients[i + 1][k];
                }
                gradients[i][j] = sum * transferFunction.calculateDerivative(neuronOutputValues[i][j]);
            }
        }
    }

    @Override
    protected boolean isBatchFull() {
        return true;
    }
}
