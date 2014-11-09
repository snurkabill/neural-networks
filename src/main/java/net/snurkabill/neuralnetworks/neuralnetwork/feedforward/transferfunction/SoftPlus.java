package net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction;

public class SoftPlus implements TransferFunctionCalculator {

    private static final double POSITIVE_ARGUMENT_LIMIT = 1.317;

    @Override
    public double getTopLimit() {
        return Double.MAX_VALUE;
    }

    @Override
    public double getLowLimit() {
        return 0;
    }

    @Override
    public double getPositiveArgumentRange() {
        return POSITIVE_ARGUMENT_LIMIT;
    }

    @Override
    public double getNegativeArgumentRange() {
        return -POSITIVE_ARGUMENT_LIMIT;
    }

    @Override
    public double calculateOutputValue(double x) {
        return Math.log(1 + Math.exp(x));
    }

    @Override
    public double calculateDerivative(double x) {
        return (1.0 / (1.0 + Math.exp(-x)));
    }

    @Override
    public String getName() {
        return null;
    }
}
