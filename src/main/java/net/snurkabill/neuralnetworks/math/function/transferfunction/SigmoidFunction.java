package net.snurkabill.neuralnetworks.math.function.transferfunction;

public class SigmoidFunction implements TransferFunctionCalculator {

    public static double TOP_LIMIT = 1.0;
    public static double BOTTOM_LIMIT = 0.0;
    public static double POSITIVE_ARGUMENT_LIMIT = 1.317;

    @Override
    public double getTopLimit() {
        return TOP_LIMIT;
    }

    @Override
    public double getLowLimit() {
        return BOTTOM_LIMIT;
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
        return (1.0 / (1.0 + Math.exp(-x)));
    }

    @Override
    public double calculateDerivative(double x) {
        double tmp = Math.exp(x);
        return tmp / ((tmp + 1.0) * (tmp + 1.0));
    }

    @Override
    public String getName() {
        return "SimpleSigmoid";
    }
}
