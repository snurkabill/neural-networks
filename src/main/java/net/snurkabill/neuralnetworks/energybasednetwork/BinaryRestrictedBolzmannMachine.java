package net.snurkabill.neuralnetworks.energybasednetwork;

import net.snurkabill.neuralnetworks.feedforwardnetwork.weightfactory.WeightsFactory;

public class BinaryRestrictedBolzmannMachine extends RestrictedBolztmannMachine {

	public BinaryRestrictedBolzmannMachine(int numOfVisible, int numOfHidden, WeightsFactory wFactory, 
			HeuristicParamsRBM heuristicParams, long seed) {
		super(numOfVisible, numOfHidden, wFactory, heuristicParams, seed);
	}

	@Override
	protected void calcVisibleNeurons() {
		for (int i = 0; i < sizeOfVisibleVector; i++) {
			visibleNeurons[i] = 
					(super.calcProbabilityOfPositiveOutput(calcVisiblePotential(i)) < random.nextDouble()) ? 1.0 : 0.0;
		}
	}

	@Override
	protected void calcHiddenNeurons() {
		for (int i = 0; i < sizeOfHiddenVector; i++) {
			hiddenNeurons[i] =
					(super.calcProbabilityOfPositiveOutput(calcHiddenPotential(i)) < random.nextDouble()) ? 1.0 : 0.0;
		}
	}

	@Override
	public double calcEnergy() {
		super.energy = 0.0;
		double helpEnergy = 0.0;
		for (int i = 0; i < sizeOfVisibleVector; i++) {
			helpEnergy += visibleNeurons[i] * visibleBias[i];
		}
		super.energy -= helpEnergy;
		helpEnergy = 0.0;
		for (int i = 0; i < sizeOfHiddenVector; i++) {
			helpEnergy += hiddenNeurons[i] * hiddenBias[i];
		}
		super.energy -= helpEnergy;
		helpEnergy = 0.0;
		for (int i = 0; i < sizeOfVisibleVector; i++) {
			for (int j = 0; j < sizeOfHiddenVector; j++) {
				helpEnergy += visibleNeurons[i] * hiddenNeurons[j] * weights[i][j];
			}
		}
		super.energy -= helpEnergy;
		return super.getEnergy();
	}
}
