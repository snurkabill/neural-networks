package net.snurkabill.neuralnetworks.managers.boltzmannmodel.validator;

import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.BoltzmannMachine;

public interface BoltzmannMachineValidator {

    public double validate(double[] inputVector, BoltzmannMachine machine);

}
