package net.snurkabill.neuralnetworks.feedforwardnetwork.trasferfunction;

public class ParametrizedSigmoidFunctionForBinaryVector implements TransferFunctionCalculator {

    public static double TOP_LIMIT = 0.5;
    public static double BOTTOM_LIMIT = -0.5;

    @Override
    public double getTopLimit() {
        return TOP_LIMIT;
    }

    @Override
    public double getLowLimit() {
        return BOTTOM_LIMIT;
    }

    @Override
    public double calculateOutputValue(double x) {
        return (1.0 / (1.0 + Math.exp(-(x)))) - 0.5;
    }

    @Override
    public double calculateDerivative(double x) {
        double tmp = Math.exp(x);
        return tmp / ((tmp + 1.0) * (tmp + 1.0));
    }

    @Override
    public String getName() {
        return "SimpleSigmoid(-0.5)";
    }
}
