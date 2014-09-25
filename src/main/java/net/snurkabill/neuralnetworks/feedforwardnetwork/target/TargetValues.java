package net.snurkabill.neuralnetworks.feedforwardnetwork.target;

import net.snurkabill.neuralnetworks.feedforwardnetwork.trasferfunction.TransferFunctionCalculator;

public abstract class TargetValues {
	
	protected final TransferFunctionCalculator transferFunctionCalculator;
	
	public TargetValues(TransferFunctionCalculator transferFunctionCalculator) {
		this.transferFunctionCalculator = transferFunctionCalculator;
	}
	
	public abstract double[] getTargetValues(int acutalIndex, int previousIndex, double[] targetValues);
	public abstract double[] getTargetValues(int actualIndex, double[] targetValues);
}
