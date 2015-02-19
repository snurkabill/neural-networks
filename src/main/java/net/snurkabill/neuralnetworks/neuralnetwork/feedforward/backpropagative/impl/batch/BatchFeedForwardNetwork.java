package net.snurkabill.neuralnetworks.neuralnetwork.feedforward.backpropagative.impl.batch;

import net.snurkabill.neuralnetworks.heuristic.FeedForwardHeuristic;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.backpropagative.BackPropagative;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction.TransferFunctionCalculator;
import net.snurkabill.neuralnetworks.weights.weightfactory.WeightsFactory;

import java.util.List;

@Deprecated // training method is wrong, batch should consists only of elements from one class
public class BatchFeedForwardNetwork extends BackPropagative {

    private final int sizeOfBatch;
    private int trainedBeforeLearning;

    public BatchFeedForwardNetwork(String name, List<Integer> topology, WeightsFactory wFactory,
                                   FeedForwardHeuristic heuristic,
                                   TransferFunctionCalculator transferFunction, int sizeOfBatch) {
        super(name, topology, wFactory, heuristic, transferFunction);
        this.sizeOfBatch = sizeOfBatch;
    }

    @Override
    protected void calcGradients(double[] targetValues) {
        for (int i = 0; i < topology[lastLayerIndex]; i++) {
            gradients[lastLayerIndex][i] += (targetValues[i] - neuronOutputValues[lastLayerIndex][i]) *
                    transferFunction.calculateDerivative(neuronOutputValues[lastLayerIndex][i]);
        }

        for (int i = numOfLayers - 2; i > 0; i--) {
            for (int j = 0; j < topology[i]; j++) {
                double sum = 0.0;
                for (int k = 0; k < topology[i + 1]; k++) {
                    sum += weights[i][j][k] * gradients[i + 1][k];
                }
                gradients[i][j] += sum * transferFunction.calculateDerivative(neuronOutputValues[i][j]);
            }
        }
        trainedBeforeLearning++;
    }

    @Override
    protected void weightsCalculation() {
        super.weightsCalculation();
        clearGradients();
        trainedBeforeLearning = 0;
    }

    private void clearGradients() {
        for (int i = lastLayerIndex; i > 0; i--) {
            for (int j = 0; j < topology[i]; j++) {
                gradients[i][j] = 0.0;
            }
        }
    }

    @Override
    protected boolean isBatchFull() {
        return trainedBeforeLearning >= sizeOfBatch;
    }

    @Override
    public String determineWorkingName() {
        return super.determineWorkingName() + " batch size: " + this.sizeOfBatch;
    }

    // TODO: conjugate gradients
}
