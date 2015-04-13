package net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.binaryvisible;

import net.snurkabill.neuralnetworks.heuristic.BoltzmannMachineHeuristic;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.weights.weightfactory.WeightsFactory;

public class BinaryToBinaryRBM extends RestrictedBoltzmannMachine {

    public BinaryToBinaryRBM(String name, int numOfVisible, int numOfHidden, WeightsFactory wFactory,
                             BoltzmannMachineHeuristic heuristicParams, long seed) {
        super(name, numOfVisible, numOfHidden, wFactory, heuristicParams, seed);
    }

    @Override
    protected void calcVisibleNeurons() {
        for (int i = 0; i < sizeOfVisibleVector; i++) {
            visibleNeurons[i] = calcProbabilityOfPositiveOutput(calcVisiblePotential(i));
            // binomial sampling
            //(super.calcProbabilityOfPositiveOutput(calcVisiblePotential(i)) > random.nextDouble()) ? 1.0 : 0.0;
        }
    }

    @Override
    protected void calcHiddenNeurons() {
        for (int i = 0; i < sizeOfHiddenVector; i++) {
            hiddenNeurons[i] = calcProbabilityOfPositiveOutput(calcHiddenPotential(i));
            // binomial sampling
            //(super.calcProbabilityOfPositiveOutput(calcHiddenPotential(i)) > random.nextDouble()) ? 1.0 : 0.0;
        }
    }

    @Override
    public double[] reconstructNext() {
        for (int i = 0; i < this.getSizeOfVisibleVector(); i++) {
            //this.getVisibleNeurons()[i] = this.random.nextInt(2);
            this.getVisibleNeurons()[i] = 0.5;
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
