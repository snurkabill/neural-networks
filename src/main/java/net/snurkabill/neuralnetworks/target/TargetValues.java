package net.snurkabill.neuralnetworks.target;

import net.snurkabill.neuralnetworks.math.function.transferfunction.TransferFunctionCalculator;

public abstract class TargetValues {

    // TODO: general evaluator!

    protected final TransferFunctionCalculator transferFunctionCalculator;
    protected final double[] targetValues;

    public TargetValues(TransferFunctionCalculator transferFunctionCalculator, int sizeOfTargetVector) {
        this.transferFunctionCalculator = transferFunctionCalculator;
        this.targetValues = new double[sizeOfTargetVector];
    }

    public double[] nullTargetValues() {
        for (int i = 0; i < targetValues.length; i++) {
            targetValues[i] = transferFunctionCalculator.getLowLimit();
        }
        return targetValues;
    }

    public abstract double[] getTargetValues(int acutalIndex, int previousIndex);

    public abstract double[] getTargetValues(int actualIndex);

    public final int lengthOfArray() {
        return targetValues.length;
    }
}
