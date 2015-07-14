package net.snurkabill.neuralnetworks.managers.supervised.regression;

import net.snurkabill.neuralnetworks.heuristic.calculators.HeuristicCalculator;
import net.snurkabill.neuralnetworks.managers.supervised.SupervisedNetworkManager;
import net.snurkabill.neuralnetworks.neuralnetwork.NeuralNetwork;
import net.snurkabill.neuralnetworks.newdata.database.NewDatabase;
import net.snurkabill.neuralnetworks.results.SupervisedNetworkResults;

public abstract class RegressionNetworkManager extends SupervisedNetworkManager {

    public RegressionNetworkManager(NeuralNetwork neuralNetwork, NewDatabase database, HeuristicCalculator heuristicCalculator, TrainingMode trainingMode) {
        super(neuralNetwork, database, heuristicCalculator, trainingMode);
    }

    @Override
    protected void processResults() {
        // TODO: add tracking of more error calculs like RMS atc ...
        super.results.add(new SupervisedNetworkResults(super.learnedVectorsBeforeTest,
                globalError, super.learningTimeBeforeTest, 0.0));
    }

    @Override
    protected void printMoreInfo() {
        // if more errors tracked, print them here.
    }
}
