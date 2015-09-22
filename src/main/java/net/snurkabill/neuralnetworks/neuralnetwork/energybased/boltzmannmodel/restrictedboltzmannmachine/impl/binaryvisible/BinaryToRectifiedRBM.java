package net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.binaryvisible;

import net.snurkabill.neuralnetworks.heuristic.BoltzmannMachineHeuristic;
import net.snurkabill.neuralnetworks.math.function.transferfunction.SigmoidFunction;
import net.snurkabill.neuralnetworks.math.function.transferfunction.SoftPlus;
import net.snurkabill.neuralnetworks.math.function.transferfunction.TransferFunctionCalculator;
import net.snurkabill.neuralnetworks.weights.weightfactory.WeightsFactory;

public class BinaryToRectifiedRBM extends BinaryToBinaryRBM {

    private final TransferFunctionCalculator hiddenTransferFunction = new SoftPlus();

    public BinaryToRectifiedRBM(String name, int numOfVisible, int numOfHidden, WeightsFactory wFactory,
                                BoltzmannMachineHeuristic heuristicParams, long seed) {
        super(name, numOfVisible, numOfHidden, wFactory, heuristicParams, seed);
    }

    @Override
    protected void calcHiddenNeurons() {
        for (int i = 0; i < hiddenNeurons.length; i++) {
            hiddenNeurons[i] = hiddenTransferFunction.calculateOutputValue(calcHiddenPotential(i));
        }
        /*for (int i = 0; i < sizeOfHiddenVector; i++) {
            hiddenNeurons[i] = hiddenTransferFunction.calculateOutputValue(
                    *//*calcProbabilityOfPositiveOutput*//*(calcHiddenPotential(i)));
        }*/
    }

    @Override
    public TransferFunctionCalculator[] getTransferFunctions() {
        return new TransferFunctionCalculator[] {new SigmoidFunction(), hiddenTransferFunction};
    }

    @Override
    public double calcEnergy(double[] inputVector) {
        this.setVisibleNeurons(inputVector);
        this.calcHiddenNeurons();
        double energy = 0.0;
        double helpEnergy = 0.0;
        for (int i = 0; i < sizeOfVisibleVector; i++) {
            helpEnergy += visibleNeurons[i] * visibleBias[i];
        }
        energy -= helpEnergy;
        helpEnergy = 0.0;
        for (int i = 0; i < sizeOfHiddenVector; i++) {
            helpEnergy += hiddenNeurons[i] * hiddenBias[i];
        }
        energy -= helpEnergy;
        helpEnergy = 0.0;
        for (int i = 0; i < sizeOfVisibleVector; i++) {
            for (int j = 0; j < sizeOfHiddenVector; j++) {
                helpEnergy += visibleNeurons[i] * hiddenNeurons[j] * weights[i][j];
            }
            helpEnergy += visibleNeurons[i] * visibleBias[i];
        }
        energy -= helpEnergy;
        return energy;
    }

    @Override
    public String determineWorkingName() {
        String workingName = this.getName();
        workingName.concat(" [" + String.valueOf(this.sizeOfVisibleVector) + " -> " +
                String.valueOf(this.sizeOfHiddenVector) + "] ");
        workingName.concat("heuristic: " + this.getHeuristic());
        return workingName;
    }

}
