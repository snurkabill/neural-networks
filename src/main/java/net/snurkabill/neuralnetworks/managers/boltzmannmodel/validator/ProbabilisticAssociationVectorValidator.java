package net.snurkabill.neuralnetworks.managers.boltzmannmodel.validator;

import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.utilities.Utilities;

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
        double[] values = new double[inputVector.length];
        for (int i = 0; i < iterations; i++) {
            machine.calculateNetwork(inputVector);
            double[] tmp = machine.getVisibleNeurons();
            for (int j = 0; j < tmp.length; j++) {
                values[j] += tmp[j];
            }
        }
        for (int i = 0; i < values.length; i++) {
            values[i] /= iterations;
        }
        return Utilities.calcError(inputVector, values);
    }
}
