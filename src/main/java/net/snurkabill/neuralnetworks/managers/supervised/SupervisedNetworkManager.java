package net.snurkabill.neuralnetworks.managers.supervised;

import net.snurkabill.neuralnetworks.heuristic.calculators.HeuristicCalculator;
import net.snurkabill.neuralnetworks.managers.NetworkManager;
import net.snurkabill.neuralnetworks.neuralnetwork.NeuralNetwork;
import net.snurkabill.neuralnetworks.newdata.database.NewDatabase;

public abstract class SupervisedNetworkManager extends NetworkManager {

    public SupervisedNetworkManager(NeuralNetwork neuralNetwork, NewDatabase database, HeuristicCalculator heuristicCalculator, TrainingMode trainingMode) {
        super(neuralNetwork, database, heuristicCalculator, trainingMode);
    }
}
