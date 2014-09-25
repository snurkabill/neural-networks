package net.snurkabill.neuralnetworks.feedforwardnetwork.trasferfunction;

public class HyperbolicTangens implements TransferFunctionCalculator {

	public static double LIMIT = 1.0;
	
	@Override
	public double calculateOutputValue(double x) {
		//return Math.tanh(x);
		
		double tmp = Math.exp(-2 * x);
		
		return (2 * (1 / (1 + tmp))) - 1;
	}

	@Override
	public double calculateDerivative(double x) {
		double tmp = (2/(Math.pow(Math.E, x) + Math.pow(Math.E, -x)));
		return (tmp * tmp);
	}

	@Override
	public double getTopLimit() {
		return LIMIT;
	}

	@Override
	public double getLowLimit() {
		return -LIMIT;
	}
}
