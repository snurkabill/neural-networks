package net.snurkabill.neuralnetworks.managers.supervised.classification;

import net.snurkabill.neuralnetworks.heuristic.calculators.HeuristicCalculator;
import net.snurkabill.neuralnetworks.managers.supervised.SupervisedNetworkManager;
import net.snurkabill.neuralnetworks.neuralnetwork.NeuralNetwork;
import net.snurkabill.neuralnetworks.newdata.database.ClassificationDatabase;
import net.snurkabill.neuralnetworks.newdata.database.NewDatabase;
import net.snurkabill.neuralnetworks.results.SupervisedNetworkResults;

public abstract class ClassificationNetworkManager extends SupervisedNetworkManager {

    protected double percentageSuccess;
    protected int correctlyClassified;
    protected int incorrectlyClassified;
    protected int[][] confusionMatrix;
    protected double[][] confusionMatrixPercentages;

    public ClassificationNetworkManager(NeuralNetwork neuralNetwork, NewDatabase database, HeuristicCalculator heuristicCalculator, TrainingMode trainingMode) {
        super(neuralNetwork, database, heuristicCalculator, trainingMode);
        confusionMatrix = new int[database.getTargetSize()][database.getTargetSize()];
        confusionMatrixPercentages = new double[database.getTargetSize()][database.getTargetSize()];
    }

    @Override
    protected void processResults() {
        super.results.add(new SupervisedNetworkResults(super.learnedVectorsBeforeTest,
                globalError, super.learningTimeBeforeTest, percentageSuccess));
    }

    @Override
    protected void printMoreInfo() {
        LOGGER.info("Percentage success: {}", percentageSuccess);
        if(LOGGER.isDebugEnabled()) {
            for (int i = 0; i < database.getTargetSize(); i++) {
                for (int j = 0; j < database.getTargetSize(); j++) {
                    confusionMatrixPercentages[i][j] = (confusionMatrix[i][j] * 100.0)
                            / ((ClassificationDatabase)database).getSizeOfValidationSetClass(i);
                }
            }
            LOGGER.debug("Confusion Matrix:");
            for (int i = 0; i < database.getTargetSize(); i++) {
                LOGGER.debug("{}: {}, {}", i, confusionMatrix[i], confusionMatrixPercentages[i]);
            }
        }
    }

    protected void nullTestingVariables() {
        this.globalError = 0.0;
        this.correctlyClassified = 0;
        this.incorrectlyClassified = 0;
        for (int i = 0; i < database.getTargetSize(); i++) {
            for (int j = 0; j < database.getTargetSize(); j++) {
                confusionMatrix[i][j] = 0;
            }
        }
    }

    protected void noteResults(int pickedClass, int actualClass) {
        if(pickedClass != Integer.MIN_VALUE) {
            confusionMatrix[actualClass][pickedClass]++;
        }
        if (pickedClass == actualClass) {
            correctlyClassified++;
        } else incorrectlyClassified++;
    }

}
