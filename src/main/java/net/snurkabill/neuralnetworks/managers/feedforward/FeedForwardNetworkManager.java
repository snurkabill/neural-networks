package net.snurkabill.neuralnetworks.managers.feedforward;

import net.snurkabill.neuralnetworks.data.database.Database;
import net.snurkabill.neuralnetworks.data.database.LabelledItem;
import net.snurkabill.neuralnetworks.heuristic.calculators.HeuristicCalculator;
import net.snurkabill.neuralnetworks.managers.NetworkManager;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.FeedForwardNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.backpropagative.BackPropagative;
import net.snurkabill.neuralnetworks.results.SupervisedNetworkResults;

public class FeedForwardNetworkManager extends NetworkManager {

    public FeedForwardNetworkManager(BackPropagative neuralNetwork, Database database,
                                     HeuristicCalculator heuristicCalculator) {
        super(neuralNetwork, database, heuristicCalculator);
    }

    @Override
    protected void train(int numOfIterations) {
        int[] learnedPatterns = new int[neuralNetwork.getSizeOfOutputVector()];
        for (int i = 0; i < numOfIterations; i++) {
            LabelledItem item = infiniteTrainingIterator.next();
            neuralNetwork.calculateNetwork(item.data);
            double[] targetValues = targetMaker.getTargetValues(item._class);
            ((BackPropagative) neuralNetwork).trainNetwork(targetValues);
            learnedPatterns[item._class]++;
        }
        if (LOGGER.isDebugEnabled()) {
            for (int i = 0; i < learnedPatterns.length; i++) {
                LOGGER.debug("Class: {}, learned patterns: {}", i, learnedPatterns[i]);
            }
        }
    }

    @Override
    protected void test() {
        FeedForwardNetwork network = (FeedForwardNetwork) super.neuralNetwork;
        double[] targetValues = targetMaker.nullTargetValues();
        int[] successValuesCounter = new int[neuralNetwork.getSizeOfOutputVector()];

        globalError = 0.0;
        int success = 0;
        int fail = 0;
        for (int _class = 0; _class < database.getNumberOfClasses(); _class++) {
            targetMaker.getTargetValues(_class);
            Database.TestClassIterator testingIterator = database.getTestingIteratorOverClass(_class);
            for (; testingIterator.hasNext(); ) {
                double[] item = testingIterator.next().data;
                globalError += network.calcError(item, targetValues);
                int pickedClass = network.getFirstHighestValueIndex();
                if(pickedClass != -1) {
                    confusionMatrix[_class][pickedClass]++;
                }
                if (pickedClass == _class) {
                    success++;
                    successValuesCounter[_class]++;
                } else fail++;
            }
        }
        int all = (success + fail);
        percentageSuccess = ((double) success * 100.0) / (double) all;
        globalError /= all;
    }

    @Override
    protected void processResults() {
        super.results.add(new SupervisedNetworkResults(super.learnedVectorsBeforeTest,
                globalError, super.learningTimeBeforeTest, percentageSuccess));
    }

    @Override
    protected void checkVectorSizes() {
        if (neuralNetwork.getSizeOfOutputVector() != database.getNumberOfClasses()) {
            throw new IllegalArgumentException("Size of output vector is different from number of classes");
        }
        if (neuralNetwork.getSizeOfInputVector() != database.getSizeOfVector()) {
            throw new IllegalArgumentException("Size of input vector is different from vectors in database");
        }
    }
}
