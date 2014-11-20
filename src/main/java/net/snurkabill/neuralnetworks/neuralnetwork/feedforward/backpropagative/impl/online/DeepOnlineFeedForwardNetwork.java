package net.snurkabill.neuralnetworks.neuralnetwork.feedforward.backpropagative.impl.online;

import net.snurkabill.neuralnetworks.heuristic.FFNNHeuristic;
import net.snurkabill.neuralnetworks.neuralnetwork.deep.DeepNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.backpropagative.BackPropagative;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction.TransferFunctionCalculator;
import net.snurkabill.neuralnetworks.weights.weightfactory.WeightsFactory;

import java.util.List;

public abstract class DeepOnlineFeedForwardNetwork extends BackPropagative implements DeepNetwork {

    public DeepOnlineFeedForwardNetwork(String name, List<Integer> topology, WeightsFactory wFactory, FFNNHeuristic heuristic, TransferFunctionCalculator transferFunction) {
        super(name, topology, wFactory, heuristic, transferFunction);
    }
}
