package net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.linearvisible;

import net.snurkabill.neuralnetworks.heuristic.BoltzmannMachineHeuristic;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.binaryvisible.BinaryToBinaryRBM;
import net.snurkabill.neuralnetworks.weights.weightfactory.WeightsFactory;

/**
 * Created by snurkabill on 14.7.15.
 */
public class LinearToLinearRBM extends BinaryToBinaryRBM {

    public LinearToLinearRBM(String name, int numOfVisible, int numOfHidden, WeightsFactory wFactory, BoltzmannMachineHeuristic heuristicParams, long seed) {
        super(name, numOfVisible, numOfHidden, wFactory, heuristicParams, seed);
    }

    @Override
    protected void calcVisibleNeurons() {
        for (int i = 0; i < sizeOfVisibleVector; i++) {
            visibleNeurons[i] = calcVisiblePotential(i);
        }
    }

    @Override
    protected void calcHiddenNeurons() {
        for (int i = 0; i < sizeOfHiddenVector; i++) {
            hiddenNeurons[i] = calcHiddenPotential(i);
        }
    }
}
