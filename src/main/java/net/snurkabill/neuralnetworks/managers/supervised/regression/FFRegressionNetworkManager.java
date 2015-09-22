package net.snurkabill.neuralnetworks.managers.supervised.regression;

import net.snurkabill.neuralnetworks.heuristic.calculators.HeuristicCalculator;
import net.snurkabill.neuralnetworks.neuralnetwork.NeuralNetwork;
import net.snurkabill.neuralnetworks.newdata.database.NewDataItem;
import net.snurkabill.neuralnetworks.newdata.database.NewDatabase;

import java.util.Enumeration;

public class FFRegressionNetworkManager extends RegressionNetworkManager {

    public FFRegressionNetworkManager(NeuralNetwork neuralNetwork, NewDatabase database, HeuristicCalculator heuristicCalculator, TrainingMode trainingMode) {
        super(neuralNetwork, database, heuristicCalculator, trainingMode);
    }

    @Override
    protected void train(int numOfIterations) {
        for (int i = 0; i < numOfIterations; i++) {
            NewDataItem item = this.dataEnumerator.nextElement();
            neuralNetwork.calculateNetwork(item.data);
            neuralNetwork.trainNetwork(item.target);
        }
    }

    @Override
    protected void test(TestingMode testingMode) {
        Enumeration<NewDataItem> enumerator = testingMode == TestingMode.TESTING ?
                database.createTestingSetEnumerator()
                : database.createValidationSetEnumerator();
        int counter = 0;
        for (; enumerator.hasMoreElements(); counter++) {
            NewDataItem item = enumerator.nextElement();
            globalError += neuralNetwork.calcError(item.data, item.target);
        }
        globalError /= counter;
    }
}
