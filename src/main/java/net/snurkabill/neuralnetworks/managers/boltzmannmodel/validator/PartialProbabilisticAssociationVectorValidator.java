package net.snurkabill.neuralnetworks.managers.boltzmannmodel.validator;

import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.utilities.Utilities;

/**
 * basically for supervised learning :)
 */
public class PartialProbabilisticAssociationVectorValidator extends ProbabilisticAssociationVectorValidator {

    private final int sizeOfUnknownPartOfVector;

    public PartialProbabilisticAssociationVectorValidator(int numberOfIterations, int sizeOfUnknownPartOfVector) {
        super(numberOfIterations);
        this.sizeOfUnknownPartOfVector = sizeOfUnknownPartOfVector;
    }

    @Override
    public double validate(double[] inputVector, RestrictedBoltzmannMachine machine) {
        double distributiveError = 0.0;
        for (int k = 0; k < iterations; k++) {
            machine.calculateNetwork(inputVector);
            // going backwards due to  unknown part of pattern is on the end
            for (int i = inputVector.length - 1, j = 0; j < sizeOfUnknownPartOfVector; i--, j++) {
                double diff = inputVector[i] - machine.getOutputValues()[i];
                distributiveError += diff * diff;
            }
        }
        return (distributiveError / iterations) * Utilities.ERROR_MODIFIER;
    }
}
