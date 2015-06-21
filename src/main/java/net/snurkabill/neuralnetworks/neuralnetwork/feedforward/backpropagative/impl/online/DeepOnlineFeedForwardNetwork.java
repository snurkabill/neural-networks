package net.snurkabill.neuralnetworks.neuralnetwork.feedforward.backpropagative.impl.online;

import net.snurkabill.neuralnetworks.heuristic.FeedForwardHeuristic;
import net.snurkabill.neuralnetworks.neuralnetwork.FeedForwardable;
import net.snurkabill.neuralnetworks.neuralnetwork.NeuralNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction.TransferFunctionCalculator;
import net.snurkabill.neuralnetworks.weights.weightfactory.WeightsFactory;

import java.util.List;

public class DeepOnlineFeedForwardNetwork extends OnlineFeedForwardNetwork {

    private final FeedForwardable inputTransformation;
    private final double[] binomialDistribution;
    private final int binomialSamplingIterations;

    public DeepOnlineFeedForwardNetwork(String name, List<Integer> topology, WeightsFactory wFactory,
                                        FeedForwardHeuristic heuristic, TransferFunctionCalculator transferFunction,
                                        FeedForwardable inputTransformation, int binomialSamplingIterations) {
        super(name, topology, wFactory, heuristic, transferFunction);
        this.inputTransformation = inputTransformation;
        this.binomialDistribution = new double[inputTransformation.getForwardedValues().length];
        this.binomialSamplingIterations = binomialSamplingIterations;
    }

    @Override
    public int getSizeOfInputVector() {
        return ((NeuralNetwork) inputTransformation).getSizeOfInputVector();
    }

    @Override
    public void feedForward(double[] inputVector) {
        //TODO: sample only if input transform is stochastic model
        //TODO: add isStochastic() for models
        for (int i = 0; i < binomialDistribution.length; i++) {
            binomialDistribution[i] = 0;
        }
        for (int i = 0; i < binomialSamplingIterations; i++) {
            inputTransformation.feedForward(inputVector);
            double[] transformedInputVector = inputTransformation.getForwardedValues();
            for (int j = 0; j < binomialDistribution.length; j++) {
                binomialDistribution[j] += transformedInputVector[j];
            }
        }
        for (int i = 0; i < binomialDistribution.length; i++) {
            binomialDistribution[i] /= binomialSamplingIterations;
            //binomialDistribution[i] = (1 - binomialDistribution[i]);
        }
        System.arraycopy(binomialDistribution, 0, neuronOutputValues[0], 0, binomialDistribution.length);
        feedForward();
    }

    @Override
    public void calculateNetwork(double[] inputVector) {
        this.feedForward(inputVector);
    }

}
