package net.snurkabill.neuralnetworks.feedforwardnetwork;

public class HyperbolicTangens implements TransferFunctionCalculator {

	@Override
	/*public double calculateOutputValue(double x) {
		return Math.tanh(x);
	}*/
	
	public double calculateOutputValue(double x) {
		double tmp = Math.pow(Math.E, 2 * x);
		return (tmp  - 1)/(tmp + 1);
	   
	}

	@Override
	public double calculateDerivative(double x) {
		double tmp = (2/(Math.pow(Math.E, x) + Math.pow(Math.E, -x)));
		return (tmp * tmp);
	}

	@Override
	public double getTopLimit() {
		return 1.0;
	}

	@Override
	public double getLowLimit() {
		return -1.0;
	}
}
