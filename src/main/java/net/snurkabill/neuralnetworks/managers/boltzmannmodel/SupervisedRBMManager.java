package net.snurkabill.neuralnetworks.managers.boltzmannmodel;

import net.snurkabill.neuralnetworks.data.database.ClassFullDatabase;
import net.snurkabill.neuralnetworks.data.database.LabelledItem;
import net.snurkabill.neuralnetworks.heuristic.calculators.HeuristicCalculator;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.validator.PartialProbabilisticAssociationVectorValidator;
import net.snurkabill.neuralnetworks.neuralnetwork.NeuralNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.results.SupervisedNetworkResults;

public class SupervisedRBMManager extends RestrictedBoltzmannMachineManager {

    private PartialProbabilisticAssociationVectorValidator validator;

    public SupervisedRBMManager(NeuralNetwork neuralNetwork, ClassFullDatabase classFullDatabase, long seed,
                                HeuristicCalculator heuristicCalculator,
                                PartialProbabilisticAssociationVectorValidator validator) {
        super(neuralNetwork, classFullDatabase, seed, heuristicCalculator);
        this.validator = validator;
    }

    @Override
    protected void train(int numOfIterations) {
        double[] inputVector = new double[classFullDatabase.getSizeOfVector() + classFullDatabase.getNumberOfClasses()];
        for (int i = 0; i < numOfIterations; i++) {
            LabelledItem item = this.infiniteTrainingIterator.next();
            System.arraycopy(item.data, 0, inputVector, 0, item.data.length);
            double[] targetValues = targetMaker.getTargetValues(item._class);
            for (int j = item.data.length, k = 0; j < inputVector.length; j++, k++) {
                inputVector[j] = targetValues[k];
            }
            neuralNetwork.trainNetwork(inputVector);
        }
    }

    @Override
    protected void test() {
        RestrictedBoltzmannMachine machine = (RestrictedBoltzmannMachine) neuralNetwork;
        int[] successValuesCounter = new int[neuralNetwork.getSizeOfOutputVector()];
        globalError = 0.0;
        int success = 0;
        int fail = 0;
        for (int _class = 0; _class < classFullDatabase.getNumberOfClasses(); _class++) {
            ClassFullDatabase.TestClassIterator testingIterator = classFullDatabase.getTestingIteratorOverClass(_class);
            for (; testingIterator.hasNext(); ) {
                double[] item = this.fillTestingVectorForReconstruction(testingIterator.next().data);
                globalError += validator.validate(item, machine);
                int pickedClass = validator.getClassWithHighestProbability();
                if(pickedClass != Integer.MIN_VALUE) {
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

    private double[] fillTestingVectorForReconstruction(double[] tmpItem) {
        double[] item = new double[classFullDatabase.getSizeOfVector() + classFullDatabase.getNumberOfClasses()];
        System.arraycopy(tmpItem, 0, item, 0, tmpItem.length);
        for (int i = 0, j = tmpItem.length; i < classFullDatabase.getNumberOfClasses(); i++, j++) {
            item[j] = 0;
        }
        return item;
    }

    @Override
    protected void processResults() {
        super.results.add(new SupervisedNetworkResults(super.learnedVectorsBeforeTest,
                globalError, super.learningTimeBeforeTest, percentageSuccess));
    }

    @Override
    protected void checkVectorSizes() {
        if (neuralNetwork.getSizeOfInputVector() != classFullDatabase.getNumberOfClasses() + classFullDatabase.getSizeOfVector()) {
            throw new IllegalArgumentException("Size of input vector is different from number of classes");
        }
    }

}
