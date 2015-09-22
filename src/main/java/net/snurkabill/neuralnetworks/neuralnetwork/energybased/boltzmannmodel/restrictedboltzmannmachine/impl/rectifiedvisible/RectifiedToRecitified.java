package net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.rectifiedvisible;

import net.snurkabill.neuralnetworks.heuristic.BoltzmannMachineHeuristic;
import net.snurkabill.neuralnetworks.math.function.transferfunction.SoftPlus;
import net.snurkabill.neuralnetworks.math.function.transferfunction.TransferFunctionCalculator;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.weights.weightfactory.WeightsFactory;

public class RectifiedToRecitified extends RestrictedBoltzmannMachine {

    private final TransferFunctionCalculator visibleTransferFunction = new SoftPlus();
    private final TransferFunctionCalculator hiddenTransferFunction = new SoftPlus();

    public RectifiedToRecitified(String name, int numOfVisible, int numOfHidden, WeightsFactory wFactory, BoltzmannMachineHeuristic heuristic, long seed) {
        super(name, numOfVisible, numOfHidden, wFactory, heuristic, seed);
    }

    @Override
    protected void calcVisibleNeurons() {
        for (int i = 0; i < visibleNeurons.length; i++) {
            visibleNeurons[i] = visibleTransferFunction.calculateOutputValue((calcVisiblePotential(i)));
        }
    }

    @Override
    protected void calcHiddenNeurons() {
        for (int i = 0; i < sizeOfHiddenVector; i++) {
            hiddenNeurons[i] = hiddenTransferFunction.calculateOutputValue((calcHiddenPotential(i)));
        }
    }

    @Override
    public double[] reconstructNext() {
        return new double[0];
    }

    @Override
    public double calcEnergy(double[] inputVector) {
        return 0;
    }

    @Override
    public TransferFunctionCalculator[] getTransferFunctions() {
        return new TransferFunctionCalculator[0];
    }
}
