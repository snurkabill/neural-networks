package net.snurkabill.neuralnetworks.math.function.transferfunction;

public class LinearFunction implements TransferFunctionCalculator {

    @Override
    public double getTopLimit() {
        return Double.MAX_VALUE;
    }

    @Override
    public double getLowLimit() {
        return Double.MIN_VALUE;
    }

    @Override
    public double getPositiveArgumentRange() {
        return 1;
    }

    @Override
    public double getNegativeArgumentRange() {
        return -1;
    }

    @Override
    public double calculateOutputValue(double x) {
        return x;
    }

    @Override
    public double calculateDerivative(double x) {
        return 1;
    }

    @Override
    public String getName() {
        return "Linear function";
    }
}
