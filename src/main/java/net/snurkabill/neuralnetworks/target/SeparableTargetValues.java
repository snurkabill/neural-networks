package net.snurkabill.neuralnetworks.target;

import net.snurkabill.neuralnetworks.math.function.transferfunction.TransferFunctionCalculator;

public class SeparableTargetValues extends TargetValues {

    public SeparableTargetValues(TransferFunctionCalculator transferFunctionCalculator, int sizeOfTargetVector) {
        super(transferFunctionCalculator, sizeOfTargetVector);
    }

    @Override
    public double[] getTargetValues(int actualIndex, int previousIndex) {
        targetValues[previousIndex] = this.transferFunctionCalculator.getLowLimit();
        targetValues[actualIndex] = this.transferFunctionCalculator.getTopLimit();
        return targetValues;
    }

    @Override
    public double[] getTargetValues(int actualIndex) {
        for (int i = 0; i < targetValues.length; i++) {
            targetValues[i] = this.transferFunctionCalculator.getLowLimit();
        }
        targetValues[actualIndex] = this.transferFunctionCalculator.getTopLimit();
        return targetValues;
    }
}
