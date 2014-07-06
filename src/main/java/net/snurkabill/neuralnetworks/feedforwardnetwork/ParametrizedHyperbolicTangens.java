package net.snurkabill.neuralnetworks.feedforwardnetwork;

public class ParametrizedHyperbolicTangens implements TransferFunctionCalculator {

	private static final double LIMIT = 1.7159;
	private static final double MODIFIER_OF_ARGUMENT = 2.0/3.0;
	private static final double DERIVATIVE_OF_LIMIT = 1.14393;
	
	@Override
	public double calculateOutputValue(double x) {
		return LIMIT * Math.tanh(MODIFIER_OF_ARGUMENT * x);
	}

	@Override
	public double calculateDerivative(double x) {
		x *= MODIFIER_OF_ARGUMENT;
		double tmp = (2/(Math.pow(Math.E, x) + Math.pow(Math.E, -x)));
		return DERIVATIVE_OF_LIMIT * (tmp * tmp);
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
