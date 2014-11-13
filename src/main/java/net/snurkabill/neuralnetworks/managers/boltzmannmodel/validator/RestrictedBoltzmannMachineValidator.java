package net.snurkabill.neuralnetworks.managers.boltzmannmodel.validator;

import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;

public interface RestrictedBoltzmannMachineValidator {

    public double validate(double[] inputVector, RestrictedBoltzmannMachine machine);

}
