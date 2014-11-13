package net.snurkabill.neuralnetworks.managers.boltzmannmodel.validator;

import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.utilities.Utilities;

/**
 * basically for supervised learning :)
 */
public class PartialProbabilisticAssociationVectorValidator extends ProbabilisticAssociationVectorValidator {

    private final int sizeOfUnknownPartOfVector;
    private final double[] sums;

    public PartialProbabilisticAssociationVectorValidator(int numberOfIterations, int sizeOfUnknownPartOfVector) {
        super(numberOfIterations);
        this.sizeOfUnknownPartOfVector = sizeOfUnknownPartOfVector;
        this.sums = new double [this.sizeOfUnknownPartOfVector];
    }

    @Override
    public double validate(double[] inputVector, RestrictedBoltzmannMachine machine) {
        double distributiveError = 0.0;
        for (int i = 0; i < sums.length; i++) {
            sums[i] = 0.0;
        }
        double[] outputValues = machine.getOutputValues();
        for (int k = 0; k < iterations; k++) {
            machine.calculateNetwork(inputVector);
            // going backwards due to  unknown part of pattern is on the end
            for (int i = inputVector.length - 1, j = 0; j < sizeOfUnknownPartOfVector; i--, j++) {
                sums[i] += outputValues[i];
                double diff = inputVector[i] - outputValues[i];
                distributiveError += diff * diff;
            }
        }
        for (int i = 0; i < sums.length; i++) {
            sums[i] /= iterations;
        }
        return (distributiveError / iterations) * Utilities.ERROR_MODIFIER;
    }

    public int getClassWithHighestProbability() {
        boolean isOnlyValueTheBiggest = false;
        int classIndex = Integer.MIN_VALUE;
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < sums.length; i++) {
            if(max == sums[i]) { // POSSIBLE COMPARISON PROBLEMS - but we don't care since this is probabilistic model
                isOnlyValueTheBiggest = false;
            } else if(max < sums[i]) {
                max = sums[i];
                classIndex = i;
                isOnlyValueTheBiggest = true;
            }
        }
        return isOnlyValueTheBiggest ? classIndex : Integer.MIN_VALUE;
    }
}
