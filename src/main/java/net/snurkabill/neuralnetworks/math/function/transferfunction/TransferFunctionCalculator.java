package net.snurkabill.neuralnetworks.math.function.transferfunction;

public interface TransferFunctionCalculator {

    double getTopLimit();

    double getLowLimit();

    double getPositiveArgumentRange();

    double getNegativeArgumentRange();

    double calculateOutputValue(double x);

    double calculateDerivative(double x);

    String getName();
}
