package net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction;

public class HyperbolicTangens implements TransferFunctionCalculator {

    public static double TOP_LIMIT = 1.0;
    public static double BOTTOM_LIMIT = -1.0;
    public static double POSITIVE_ARGUMENT_LIMIT = 0.658;

    @Override
    public double calculateOutputValue(double x) {
        //return Math.tanh(x);
        double tmp = Math.exp(-2 * x);
        return (2 * (1 / (1 + tmp))) - 1; // exactly same as Math.tanh(x), should be faster;
    }

    @Override
    public double calculateDerivative(double x) {
        double tmp = (2 / (Math.pow(Math.E, x) + Math.pow(Math.E, -x)));
        return (tmp * tmp);
    }

    @Override
    public double getTopLimit() {
        return TOP_LIMIT;
    }

    @Override
    public double getLowLimit() {
        return -BOTTOM_LIMIT;
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
    public String getName() {
        return "HyperbolicTangens";
    }
}
