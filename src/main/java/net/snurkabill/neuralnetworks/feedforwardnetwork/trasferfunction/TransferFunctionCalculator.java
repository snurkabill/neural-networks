package net.snurkabill.neuralnetworks.feedforwardnetwork.trasferfunction;

public interface TransferFunctionCalculator {
	
	public double getTopLimit();
	public double getLowLimit();
	
	public double calculateOutputValue(double x);
	public double calculateDerivative(double x);

    public String getName();
}
