package net.snurkabill.neuralnetworks.neuralnetwork.feedforward.backpropagative.impl.online;

import net.snurkabill.neuralnetworks.heuristic.FFNNHeuristic;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.FeedForwardNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction.TransferFunctionCalculator;
import net.snurkabill.neuralnetworks.weights.weightfactory.WeightsFactory;

import java.util.List;

public class DeepOnlineFeedForwardNetwork extends OnlineFeedForwardNetwork {

    private final FeedForwardNetwork inputTransformation;

    public DeepOnlineFeedForwardNetwork(String name, List<Integer> topology, WeightsFactory wFactory,
                                        FFNNHeuristic heuristic, TransferFunctionCalculator transferFunction,
                                        FeedForwardNetwork inputTransformation) {
        super(name, topology, wFactory, heuristic, transferFunction);
        this.inputTransformation = inputTransformation;
    }

    @Override
    public void feedForward(double[] inputVector) {
        inputTransformation.feedForward(inputVector);
        double[] transformedInputVector = inputTransformation.getOutputValues();
        System.arraycopy(transformedInputVector, 0, neuronOutputValues[0], 0, transformedInputVector.length);
        feedForward();
    }

    @Override
    public void calculateNetwork(double[] inputVector) {
        this.feedForward(inputVector);
    }

}
