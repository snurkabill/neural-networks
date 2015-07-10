package net.snurkabill.neuralnetworks.math.function.transferfunction;

public class ParametrizedHyperbolicTangens implements TransferFunctionCalculator {

    private static final double LIMIT = 1.7159;
    private static final double MODIFIER_OF_ARGUMENT = 0.6666666;
    private static final double DERIVATIVE_OF_LIMIT = 1.14393;
    private static final double POSITIVE_ARGUMENT_LIMIT = 0.988;

    @Override
    public double calculateOutputValue(double x) {

        double tmp = Math.exp(-2 * (MODIFIER_OF_ARGUMENT * x));
        return LIMIT * ((2 * (1 / (1 + tmp))) - 1); // exactly same as Math.tanh(x), should be faster;

        //return LIMIT * Math.tanh(MODIFIER_OF_ARGUMENT * x);
    }

    @Override
    public double calculateDerivative(double x) {
        x *= MODIFIER_OF_ARGUMENT;
        double tmp = (2 / (Math.pow(Math.E, x) + Math.pow(Math.E, -x)));
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
        return "ParamHyperTan(1.7159)";
    }
}
