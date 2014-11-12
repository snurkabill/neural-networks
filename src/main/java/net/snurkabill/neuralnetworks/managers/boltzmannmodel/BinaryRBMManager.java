package net.snurkabill.neuralnetworks.managers.boltzmannmodel;

import net.snurkabill.neuralnetworks.data.database.DataItem;
import net.snurkabill.neuralnetworks.data.database.Database;
import net.snurkabill.neuralnetworks.data.database.LabelledItem;
import net.snurkabill.neuralnetworks.heuristic.calculators.HeuristicCalculator;
import net.snurkabill.neuralnetworks.neuralnetwork.NeuralNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.results.SupervisedNetworkResults;
import net.snurkabill.neuralnetworks.utilities.Utilities;

import java.util.Iterator;

public class BinaryRBMManager extends RestrictedBoltzmannMachineManager {

    private double avgRightPercentageOfUnknownPart;
    private double percentageSizeOfUnknownSize;
    private int maximumAttemptsDuringTesting;
    private int minimumAttemptsDuringTesting;

    public BinaryRBMManager(NeuralNetwork neuralNetwork, Database database, long seed, int maximumAttemptsDuringTesting,
                            int minimumAttemptsDuringTesting, HeuristicCalculator heuristicCalculator) {
        super(neuralNetwork, database, seed, heuristicCalculator);
        this.maximumAttemptsDuringTesting = maximumAttemptsDuringTesting;
        this.minimumAttemptsDuringTesting = minimumAttemptsDuringTesting;
    }

    @Override
    protected void train(int numOfIterations) {
        double[] inputVector = new double[database.getSizeOfVector() + database.getNumberOfClasses()];
        for (int i = 0; i < numOfIterations; i++) {
            LabelledItem item = this.infiniteTrainingIterator.next();
            for (int j = 0; j < item.data.length; j++) {
                inputVector[j] = (item.data[j] < 30 ? 0 : 1);
            }
            double[] targetValues = targetMaker.getTargetValues(item._class);
            for (int j = item.data.length, k = 0; j < inputVector.length; j++, k++) {
                inputVector[j] = targetValues[k];
            }
            neuralNetwork.trainNetwork(inputVector);
        }
    }

    @Override
    protected void test() {
        int numOfUnknownDimensionsOfVector = database.getNumberOfClasses();
        if (numOfUnknownDimensionsOfVector >= database.getSizeOfVector()) {
            throw new IllegalArgumentException("Vector must have some known dimensions for association");
        }
        if (numOfUnknownDimensionsOfVector < 0) {
            throw new IllegalArgumentException("There can't be negative num of unknown dimensions of vector");
        }
        RestrictedBoltzmannMachine machine = (RestrictedBoltzmannMachine) neuralNetwork;
        double distributedSum = 0.0;
        double[] targetValues = targetMaker.nullTargetValues();
        int[] successValuesCounter = new int[neuralNetwork.getSizeOfOutputVector()];
        globalError = 0.0;
        int success = 0;
        int fail = 0;
        for (int _class = 0; _class < database.getNumberOfClasses(); _class++) {
            targetMaker.getTargetValues(_class);
            Iterator<DataItem> testingIterator = database.getTestingIteratorOverClass(_class);
            Integer index = Integer.MIN_VALUE;
            for (; testingIterator.hasNext(); ) {
                double[] item = this.fillTestingVectorForReconstruction(testingIterator.next().data);
                double[] results = new double[database.getNumberOfClasses()];
                for (int i = 0; i < maximumAttemptsDuringTesting; i++) {
                    machine.calculateNetwork(item);
                    double[] outputVector = machine.getOutputValues();
                    for (int j = 0; j < results.length; j++) {
                        results[j] += outputVector[database.getSizeOfVector() + j];
                    }
                    if ((index = isOnlyValueTheBiggest(results)) >= 0 && i >= minimumAttemptsDuringTesting) {
                        break;
                    }
                }
                globalError += Utilities.calcError(targetValues, results);
                if (index == _class) {
                    success++;
                    successValuesCounter[_class]++;
                } else fail++;
            }
        }
        int all = (success + fail);
        percentageSuccess = ((double) success * 100.0) / (double) all;
        globalError /= all;
    }

    private double[] fillTestingVectorForReconstruction(double[] tmpItem) {
        double[] item = new double[database.getSizeOfVector() + database.getNumberOfClasses()];
        for (int i = 0; i < tmpItem.length; i++) {
            item[i] = (tmpItem[i] < 30 ? 0 : 1);
        }
        for (int i = 0, j = tmpItem.length; i < database.getNumberOfClasses(); i++, j++) {
            item[j] = 0;
        }
        return item;
    }

    private int isOnlyValueTheBiggest(double[] values) {
        boolean isOnlyValueTheBiggest = false;
        int highestIndex = Integer.MIN_VALUE;
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < values.length; i++) {
            if (max == values[i]) { // POSSIBLE COMPARE PROBLEMS
                isOnlyValueTheBiggest = false;
            } else if (max < values[i]) {
                max = values[i];
                highestIndex = i;
                isOnlyValueTheBiggest = true;
            }
        }
        return isOnlyValueTheBiggest ? highestIndex : Integer.MIN_VALUE;
    }

    @Deprecated
    public int getFirstHighestValueIndex(double[] values) {
        double max = -1;
        int bestIndex = -1;
        for (int i = 0; i < values.length; i++) {
            if (values[i] > max) {
                max = values[i];
                bestIndex = i;
            }
        }
        return bestIndex;
    }

    @Override
    protected void processResults() {
        super.results.add(new SupervisedNetworkResults(super.learnedVectorsBeforeTest,
                globalError, super.learningTimeBeforeTest, percentageSuccess));
    }

}
