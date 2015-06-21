package net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.gaussianvisible;

import net.snurkabill.neuralnetworks.heuristic.BoltzmannMachineHeuristic;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.weights.weightfactory.WeightsFactory;

public class GaussianToBinaryRBM extends RestrictedBoltzmannMachine {

    private final double inputDeviations[];

    public GaussianToBinaryRBM(String name, int numOfVisible, int numOfHidden, WeightsFactory wFactory, BoltzmannMachineHeuristic heuristic, long seed) {
        super(name, numOfVisible, numOfHidden, wFactory, heuristic, seed);
        this.inputDeviations = new double[numOfVisible];
        for (int i = 0; i < inputDeviations.length; i++) {
            inputDeviations[i] = 1;
        }
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
            hiddenNeurons[i] =
                    (super.calcProbabilityOfPositiveOutput(calcHiddenPotential(i)) > random.nextDouble()) ? 1.0 : 0.0;
        }
    }

    @Override
    public double[] reconstructNext() {
        for (int i = 0; i < this.getSizeOfVisibleVector(); i++) {
            this.getVisibleNeurons()[i] = this.random.nextDouble();
        }
        calcHiddenNeurons();
        return activateVisibleNeurons();
    }

    @Override
    public double calcEnergy(double[] inputVector) {
        this.setVisibleNeurons(inputVector);
        this.calcHiddenNeurons();
        double energy = 0.0;
        double helpEnergy = 0.0;
        double errorModifier = 2;
        for (int i = 0; i < sizeOfVisibleVector; i++) {
            double tmp = (visibleNeurons[i] - visibleBias[i]);
            helpEnergy += (tmp * tmp) / (errorModifier * Math.pow(inputDeviations[i], 2)); // square of std. dev.
        }
        energy += helpEnergy; // yes, here is plus
        helpEnergy = 0.0;
        for (int i = 0; i < sizeOfHiddenVector; i++) {
            helpEnergy += hiddenNeurons[i] * hiddenBias[i];
        }
        energy -= helpEnergy;
        helpEnergy = 0.0;
        for (int i = 0; i < sizeOfVisibleVector; i++) {
            for (int j = 0; j < sizeOfHiddenVector; j++) {
                helpEnergy += (visibleNeurons[i] / inputDeviations[i]) * hiddenNeurons[j] * weights[i][j];
            }
            helpEnergy += visibleNeurons[i] * visibleBias[i];
        }
        energy -= helpEnergy;
        return energy;
    }
}
