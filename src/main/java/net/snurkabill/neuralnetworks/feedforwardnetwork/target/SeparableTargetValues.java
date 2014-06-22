package net.snurkabill.neuralnetworks.feedforwardnetwork.target;

import net.snurkabill.neuralnetworks.feedforwardnetwork.TransferFunctionCalculator;

public class SeparableTargetValues extends TargetValues {

	public SeparableTargetValues(TransferFunctionCalculator transferFunctionCalculator) {
		super(transferFunctionCalculator);
	}

	@Override
	public double[] getTargetValues(int actualIndex, int previousIndex, double[] targetValues) {
		
		targetValues[previousIndex] = this.transferFunctionCalculator.getLowLimit();
		targetValues[actualIndex] = this.transferFunctionCalculator.getTopLimit();
		
		return targetValues;
	}
}
