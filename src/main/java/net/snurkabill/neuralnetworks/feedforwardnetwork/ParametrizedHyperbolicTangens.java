package net.snurkabill.neuralnetworks.feedforwardnetwork;

public class ParametrizedHyperbolicTangens implements TransferFunctionCalculator {

	@Override
	public double calculateOutputValue(double x) {
		return 1.7159 * Math.tanh((2.0/3.0) * x);
	}

	@Override
	public double calculateDerivative(double x) {
		x = x * 2/3;
		double tmp = (2/(Math.pow(Math.E, x) + Math.pow(Math.E, -x)));
		return 1.14393 * (tmp * tmp);
	}

	@Override
	public double getTopLimit() {
		return 1.7159;
	}

	@Override
	public double getLowLimit() {
		return -1.7159;
	}
}
