package net.snurkabill.neuralnetworks.managers.boltzmannmodel.validator;

import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;

public class ProbabilisticAssociationVectorValidator implements RestrictedBoltzmannMachineValidator {

    protected int iterations;

    public ProbabilisticAssociationVectorValidator(int numberOfIterations) {
        this.iterations = numberOfIterations;
    }

    public int getIterations() {
        return iterations;
    }

    public void setIterations(int iterations) {
        this.iterations = iterations;
    }

    @Override
    public double validate(double[] inputVector, RestrictedBoltzmannMachine machine) {
        double distributiveError = 0.0;
        for (int i = 0; i < iterations; i++) {
            distributiveError += machine.calcError(inputVector, inputVector);
        }
        distributiveError /= iterations;
        return distributiveError;
    }
}
