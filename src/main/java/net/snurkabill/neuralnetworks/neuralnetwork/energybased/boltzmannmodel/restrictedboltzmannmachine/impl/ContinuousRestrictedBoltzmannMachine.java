package net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl;

import net.snurkabill.neuralnetworks.heuristic.HeuristicRBM;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.weights.weightfactory.WeightsFactory;

public class ContinuousRestrictedBoltzmannMachine extends BinaryRestrictedBoltzmannMachine {

    public ContinuousRestrictedBoltzmannMachine(String name, int numOfVisible, int numOfHidden, WeightsFactory wFactory,
                                                HeuristicRBM heuristic, long seed) {
        super(name, numOfVisible, numOfHidden, wFactory, heuristic, seed);
    }

    @Override
    protected void calcHiddenNeurons() {
        for (int i = 0; i < sizeOfHiddenVector; i++) {
            hiddenNeurons[i] = Math.log(1 + calcHiddenPotential(i));
        }
    }

}
