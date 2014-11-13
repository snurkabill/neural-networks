package net.snurkabill.neuralnetworks.managers.boltzmannmodel.binary;

import net.snurkabill.neuralnetworks.data.database.DataItem;
import net.snurkabill.neuralnetworks.data.database.Database;
import net.snurkabill.neuralnetworks.data.database.LabelledItem;
import net.snurkabill.neuralnetworks.heuristic.calculators.HeuristicCalculator;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.RestrictedBoltzmannMachineManager;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.validator.PartialProbabilisticAssociationVectorValidator;
import net.snurkabill.neuralnetworks.neuralnetwork.NeuralNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.results.SupervisedNetworkResults;
import net.snurkabill.neuralnetworks.utilities.Utilities;

import java.util.Iterator;

public class SupervisedBinaryRBMManager extends RestrictedBoltzmannMachineManager {

    private double avgRightPercentageOfUnknownPart;
    private double percentageSizeOfUnknownSize;
    private PartialProbabilisticAssociationVectorValidator validator;

    public SupervisedBinaryRBMManager(NeuralNetwork neuralNetwork, Database database, long seed,
                                      HeuristicCalculator heuristicCalculator,
                                      PartialProbabilisticAssociationVectorValidator validator) {
        super(neuralNetwork, database, seed, heuristicCalculator);
        this.validator = validator;
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
        RestrictedBoltzmannMachine machine = (RestrictedBoltzmannMachine) neuralNetwork;
        int[] successValuesCounter = new int[neuralNetwork.getSizeOfOutputVector()];
        globalError = 0.0;
        int success = 0;
        int fail = 0;
        for (int _class = 0; _class < database.getNumberOfClasses(); _class++) {
            targetMaker.getTargetValues(_class);
            Iterator<DataItem> testingIterator = database.getTestingIteratorOverClass(_class);
            for (; testingIterator.hasNext(); ) {
                double[] item = this.fillTestingVectorForReconstruction(testingIterator.next().data);
                globalError += validator.validate(item, machine);
                if (validator.getClassWithHighestProbability() == _class) {
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

    @Override
    protected void processResults() {
        super.results.add(new SupervisedNetworkResults(super.learnedVectorsBeforeTest,
                globalError, super.learningTimeBeforeTest, percentageSuccess));
    }

}
