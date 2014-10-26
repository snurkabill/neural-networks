package net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction;

public interface TransferFunctionCalculator {

    public double getTopLimit();

    public double getLowLimit();

    public double getPositiveArgumentRange();

    public double getNegativeArgumentRange();

    public double calculateOutputValue(double x);

    public double calculateDerivative(double x);

    public String getName();
}
