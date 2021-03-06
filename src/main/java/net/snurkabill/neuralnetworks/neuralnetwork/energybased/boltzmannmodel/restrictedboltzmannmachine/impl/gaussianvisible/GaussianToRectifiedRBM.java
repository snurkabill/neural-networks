package net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.gaussianvisible;

import net.snurkabill.neuralnetworks.heuristic.BoltzmannMachineHeuristic;
import net.snurkabill.neuralnetworks.math.function.transferfunction.LinearFunction;
import net.snurkabill.neuralnetworks.math.function.transferfunction.SoftPlus;
import net.snurkabill.neuralnetworks.math.function.transferfunction.TransferFunctionCalculator;
import net.snurkabill.neuralnetworks.weights.weightfactory.WeightsFactory;

public class GaussianToRectifiedRBM extends GaussianToBinaryRBM {

    private final TransferFunctionCalculator hiddenTransferFunction = new SoftPlus();

    public GaussianToRectifiedRBM(String name, int numOfVisible, int numOfHidden, WeightsFactory wFactory, BoltzmannMachineHeuristic heuristic, long seed) {
        super(name, numOfVisible, numOfHidden, wFactory, heuristic, seed);
    }

    @Override
    protected void calcHiddenNeurons() {
        for (int i = 0; i < sizeOfHiddenVector; i++) {
            hiddenNeurons[i] = hiddenTransferFunction.calculateOutputValue(
                    /*calcProbabilityOfPositiveOutput*/(calcHiddenPotential(i)));
        }
    }

    @Override
    public TransferFunctionCalculator[] getTransferFunctions() {
        return new TransferFunctionCalculator[] {new LinearFunction(), hiddenTransferFunction};
    }
}
