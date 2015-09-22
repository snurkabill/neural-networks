package net.snurkabill.neuralnetworks.managers.supervised.classification;

import net.snurkabill.neuralnetworks.heuristic.calculators.HeuristicCalculator;
import net.snurkabill.neuralnetworks.neuralnetwork.NeuralNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.FeedForwardNetwork;
import net.snurkabill.neuralnetworks.newdata.database.ClassificationDatabase;
import net.snurkabill.neuralnetworks.newdata.database.NewDataItem;
import net.snurkabill.neuralnetworks.newdata.database.NewDatabase;

import java.util.Enumeration;

public class FFClassificationNetworkManager extends ClassificationNetworkManager {

    public FFClassificationNetworkManager(NeuralNetwork neuralNetwork, NewDatabase database, HeuristicCalculator heuristicCalculator, TrainingMode trainingMode) {
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
        nullTestingVariables();
        for (int _class = 0; _class < database.getTargetSize(); _class++) {
            Enumeration<NewDataItem> enumerator = testingMode == TestingMode.TESTING ?
                    ((ClassificationDatabase)database).createTestClassEnumerator(_class)
                    : ((ClassificationDatabase)database).createValidationClassEnumerator(_class);
            for (; enumerator.hasMoreElements(); ) {
                NewDataItem item = enumerator.nextElement();
                globalError += neuralNetwork.calcError(item.data, item.target);
                int pickedClass = ((FeedForwardNetwork)neuralNetwork).getFirstHighestValueIndex();
                noteResults(pickedClass, _class);
            }
        }
        int all = (correctlyClassified + incorrectlyClassified);
        percentageSuccess = ((double) correctlyClassified * 100.0) / (double) all;
        globalError /= all;
    }

}
