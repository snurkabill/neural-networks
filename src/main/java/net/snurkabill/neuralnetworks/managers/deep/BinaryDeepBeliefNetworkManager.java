package net.snurkabill.neuralnetworks.managers.deep;

import net.snurkabill.neuralnetworks.data.database.DataItem;
import net.snurkabill.neuralnetworks.data.database.Database;
import net.snurkabill.neuralnetworks.data.database.LabelledItem;
import net.snurkabill.neuralnetworks.heuristic.calculators.HeuristicCalculator;
import net.snurkabill.neuralnetworks.managers.NetworkManager;
import net.snurkabill.neuralnetworks.neuralnetwork.NeuralNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.results.SupervisedNetworkResults;
import net.snurkabill.neuralnetworks.utilities.Utilities;

import java.util.Iterator;

public class BinaryDeepBeliefNetworkManager extends NetworkManager {

    public BinaryDeepBeliefNetworkManager(NeuralNetwork neuralNetwork, Database database, HeuristicCalculator heuristicCalculator) {
        super(neuralNetwork, database, heuristicCalculator);
    }

    @Override
    protected void train(int numOfIterations) {
        for (int i = 0; i < numOfIterations; i++) {
            LabelledItem item = this.infiniteTrainingIterator.next();
            double[] targetValues = targetMaker.getTargetValues(item._class);
            neuralNetwork.calculateNetwork(item.data);
            neuralNetwork.trainNetwork(targetValues);
        }
    }

    @Override
    protected void test() {
        double[] targetValues = targetMaker.nullTargetValues();
        RestrictedBoltzmannMachine machine = (RestrictedBoltzmannMachine) neuralNetwork;
        int[] successValuesCounter = new int[neuralNetwork.getSizeOfOutputVector()];
        globalError = 0.0;
        int success = 0;
        int fail = 0;
        for (int _class = 0; _class < database.getNumberOfClasses(); _class++) {
            targetMaker.getTargetValues(_class);
            Iterator<DataItem> testingIterator = database.getTestingIteratorOverClass(_class);
            for (; testingIterator.hasNext(); ) {
                DataItem item = testingIterator.next();
                neuralNetwork.calculateNetwork(item.data);
                //LOGGER.info("{}", neuralNetwork.getOutputValues());
                globalError += Utilities.calcError(targetValues, neuralNetwork.getOutputValues());
                if (getClassWithHighestProbability(neuralNetwork.getOutputValues()) == _class) {
                    success++;
                    successValuesCounter[_class]++;
                } else fail++;
            }
        }
        int all = (success + fail);
        percentageSuccess = ((double) success * 100.0) / (double) all;
        globalError /= all;
    }

    public int getClassWithHighestProbability(double[] sums) {
        boolean isOnlyValueTheBiggest = false;
        int classIndex = Integer.MIN_VALUE;
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < sums.length; i++) {
            if (max == sums[i]) { // POSSIBLE COMPARISON PROBLEMS - but we don't care since this is probabilistic model
                isOnlyValueTheBiggest = false;
            } else if (max < sums[i]) {
                max = sums[i];
                classIndex = i;
                isOnlyValueTheBiggest = true;
            }
        }
        return isOnlyValueTheBiggest ? classIndex : Integer.MIN_VALUE;
    }

    @Override
    protected void processResults() {
        super.results.add(new SupervisedNetworkResults(super.learnedVectorsBeforeTest,
                globalError, super.learningTimeBeforeTest, percentageSuccess));
    }

    @Override
    protected void checkVectorSizes() {
        if (neuralNetwork.getSizeOfInputVector() != database.getSizeOfVector()) {
            throw new IllegalArgumentException("Size of input vector is different from number of classes");
        }
        if (neuralNetwork.getSizeOfOutputVector() != database.getNumberOfClasses()) {
            throw new IllegalArgumentException("Size of output vector is different from number of classes");
        }
    }
}
