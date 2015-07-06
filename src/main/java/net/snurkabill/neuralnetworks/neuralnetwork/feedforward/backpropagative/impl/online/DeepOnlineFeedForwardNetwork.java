package net.snurkabill.neuralnetworks.neuralnetwork.feedforward.backpropagative.impl.online;

import net.snurkabill.neuralnetworks.heuristic.FeedForwardHeuristic;
import net.snurkabill.neuralnetworks.neuralnetwork.FeedForwardable;
import net.snurkabill.neuralnetworks.neuralnetwork.NeuralNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.deep.RBMCone;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction.TransferFunctionCalculator;
import net.snurkabill.neuralnetworks.weights.weightfactory.WeightsFactory;

import java.util.List;

public class DeepOnlineFeedForwardNetwork extends OnlineFeedForwardNetwork {

    private final FeedForwardable inputTransformation;

    public DeepOnlineFeedForwardNetwork(String name, List<Integer> topology, WeightsFactory wFactory,
                                        FeedForwardHeuristic heuristic, TransferFunctionCalculator transferFunction,
                                        FeedForwardable inputTransformation) {
        super(name, topology, wFactory, heuristic, transferFunction);
        this.inputTransformation = inputTransformation;
    }

    @Override
    public int getSizeOfInputVector() {
        if(inputTransformation instanceof NeuralNetwork) {
            return ((NeuralNetwork) inputTransformation).getSizeOfInputVector();
        } else if (inputTransformation instanceof RBMCone) {
            return ((RBMCone) inputTransformation).getSizeOfInputVector();
        } else {
            throw new IllegalStateException("There is no other network");
        }
    }

    @Override
    public void feedForward(double[] inputVector) {
        inputTransformation.feedForward(inputVector);
        double[] transformedInputVector = inputTransformation.getForwardedValues();
        System.arraycopy(transformedInputVector, 0, neuronOutputValues[0], 0, transformedInputVector.length);
        feedForward();
    }

    @Override
    public void calculateNetwork(double[] inputVector) {
        this.feedForward(inputVector);
    }

}
