package net.snurkabill.neuralnetworks.managers.unsupervised;

import net.snurkabill.neuralnetworks.heuristic.calculators.HeuristicCalculator;
import net.snurkabill.neuralnetworks.managers.NetworkManager;
import net.snurkabill.neuralnetworks.neuralnetwork.NeuralNetwork;
import net.snurkabill.neuralnetworks.newdata.database.NewDataItem;
import net.snurkabill.neuralnetworks.newdata.database.NewDatabase;
import net.snurkabill.neuralnetworks.results.UnsupervisedNetworkResults;

import java.util.Enumeration;

public class UnsupervisedNetworkManager extends NetworkManager {

    public UnsupervisedNetworkManager(NeuralNetwork neuralNetwork, NewDatabase database, HeuristicCalculator heuristicCalculator, TrainingMode trainingMode) {
        super(neuralNetwork, database, heuristicCalculator, trainingMode);
    }

    @Override
    protected void train(int numOfIterations) {
        for (int i = 0; i < numOfIterations; i++) {
            NewDataItem item = this.dataEnumerator.nextElement();
            neuralNetwork.trainNetwork(item.data);
        }
    }

    @Override
    protected void test(TestingMode testingMode) {
        globalError = 0.0;
        int all = 0;
        Enumeration<NewDataItem> enumerator = testingMode == TestingMode.TESTING ? database.createTestingSetEnumerator()
                : database.createValidationSetEnumerator();
        for (int i = 0; enumerator.hasMoreElements(); i++) {
            NewDataItem item = enumerator.nextElement();
            globalError += neuralNetwork.calcError(item.data, item.data);
        }
        globalError /= all;
    }

    @Override
    protected void processResults() {
        super.results.add(new UnsupervisedNetworkResults(super.learnedVectorsBeforeTest,
                globalError, super.learningTimeBeforeTest, 0, 0));
    }

    @Override
    protected void printMoreInfo() {

    }
}
