package net.snurkabill.neuralnetworks.energybasednetwork;

import net.snurkabill.neuralnetworks.feedforwardnetwork.weightfactory.WeightsFactory;

public class GeneralizedRestrictedBoltzmannMachine extends RestrictedBoltzmannMachine {

	public GeneralizedRestrictedBoltzmannMachine(int numOfVisible, int numOfHidden, WeightsFactory wFactory, HeuristicParamsRBM heuristicParams, long seed) {
		super(numOfVisible, numOfHidden, wFactory, heuristicParams, seed);
	}

	@Override
	protected void calcVisibleNeurons() {
		
	}

	@Override
	protected void calcHiddenNeurons() {
		
	}

	@Override
	public double calcEnergy() {
		throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
	}
	
}
