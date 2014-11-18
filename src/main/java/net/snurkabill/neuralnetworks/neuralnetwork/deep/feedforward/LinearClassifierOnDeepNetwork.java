package net.snurkabill.neuralnetworks.neuralnetwork.deep.feedforward;

import net.snurkabill.neuralnetworks.heuristic.FFNNHeuristic;
import net.snurkabill.neuralnetworks.neuralnetwork.deep.DeepNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.FeedForwardableNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.backpropagative.impl.online.OnlineFeedForwardNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction.TransferFunctionCalculator;
import net.snurkabill.neuralnetworks.weights.weightfactory.WeightsFactory;

import java.util.List;

public class LinearClassifierOnDeepNetwork extends OnlineFeedForwardNetwork implements DeepNetwork {

    private final FeedForwardableNetwork inputTransformation;

    public LinearClassifierOnDeepNetwork(String name, List<Integer> topology, WeightsFactory wFactory, FFNNHeuristic heuristic,
                                         TransferFunctionCalculator transferFunction, FeedForwardableNetwork inputTransformation) {
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
