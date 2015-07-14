package net.snurkabill.neuralnetworks.managers.supervised.classification;

import net.snurkabill.neuralnetworks.heuristic.calculators.HeuristicCalculator;
import net.snurkabill.neuralnetworks.neuralnetwork.NeuralNetwork;
import net.snurkabill.neuralnetworks.newdata.database.ClassificationDatabase;
import net.snurkabill.neuralnetworks.newdata.database.NewDataItem;
import net.snurkabill.neuralnetworks.newdata.database.NewDatabase;

import java.util.Enumeration;

public class RBMClassificationNetworkManager extends ClassificationNetworkManager {

    private final double[] tmpArray;
    private final double[] tmpArray2;
    private final int targetIndex;

    public RBMClassificationNetworkManager(NeuralNetwork neuralNetwork, NewDatabase database, HeuristicCalculator heuristicCalculator, TrainingMode trainingMode) {
        super(neuralNetwork, database, heuristicCalculator, trainingMode);
        this.tmpArray = new double[database.getTargetSize() + database.getVectorSize()];
        this.tmpArray2 = new double[database.getTargetSize() + database.getVectorSize()];
        this.targetIndex = database.getVectorSize();
    }

    @Override
    protected void train(int numOfIterations) {
        for (int i = 0; i < numOfIterations; i++) {
            NewDataItem item = this.dataEnumerator.nextElement();
            System.arraycopy(item.data, 0, tmpArray, 0, item.data.length);
            System.arraycopy(item.target, 0, tmpArray, targetIndex, item.target.length);
            neuralNetwork.trainNetwork(tmpArray);
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
                System.arraycopy(item.data, 0, tmpArray, 0, item.data.length);
                System.arraycopy(item.data, 0, tmpArray2, 0, item.data.length);
                for (int i = 0; i < database.getTargetSize(); i++) {
                    tmpArray[targetIndex + i] = neuralNetwork.getTransferFunction().getLowLimit();
                    tmpArray2[targetIndex + i] = item.target[i];
                }
                globalError += neuralNetwork.calcError(tmpArray, tmpArray2);
                neuralNetwork.getOutputValues(tmpArray);
                int pickedClass = this.getClassWithHighestProbability(tmpArray);
                noteResults(pickedClass, _class);
            }
        }
        int all = (correctlyClassified + incorrectlyClassified);
        percentageSuccess = ((double) correctlyClassified * 100.0) / (double) all;
        globalError /= all;
    }

    @Deprecated
    protected int getClassWithHighestProbability(double[] array) {
        boolean isOnlyValueTheBiggest = false;
        int classIndex = Integer.MIN_VALUE;
        double max = Double.NEGATIVE_INFINITY;
        for (int i = targetIndex, j = 0; j < database.getTargetSize(); i++, j++) {
            if (max == array[i]) { // POSSIBLE COMPARISON PROBLEMS - but we don't care since this is probabilistic model
                isOnlyValueTheBiggest = false;
            } else if (max < array[i]) {
                max = array[i];
                classIndex = j;
                isOnlyValueTheBiggest = true;
            }
        }
        return isOnlyValueTheBiggest ? classIndex : Integer.MIN_VALUE;
    }

}
