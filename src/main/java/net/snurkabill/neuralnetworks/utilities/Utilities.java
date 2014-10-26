package net.snurkabill.neuralnetworks.utilities;

public class Utilities {

    public static double mean(final double[] input) {
        double m = 0.0;
        for (int i = 0; i < input.length; i++)
            m += input[i];
        return m / input.length;
    }

    public static double stddev(final double[] input, double mean) {
        double sum = 0.0;
        for (int i = 0; i < input.length; i++)
            sum += (input[i] - mean) * (input[i] - mean);
        return Math.sqrt(sum / (input.length - 1));
    }

    public static double sigmoid(double x) {
        return (1.0 / (1.0 + Math.exp(-x)));
    }

    public static double sigmoidDerivative(double x) {
        double tmp = Math.exp(x);
        return tmp / ((tmp + 1.0) * (tmp + 1.0));
    }

    public static double calcError(double[] expectedValues, double[] realValues) {
        if(expectedValues.length != realValues.length) {
            throw new IllegalArgumentException("Lengths of arrays are different!");
        }
        double error = 0.0;
        for (int i = 0; i < expectedValues.length; i++) {
            double diff = expectedValues[i] - realValues[i];
            error += diff * diff;
        }
        return 0.5 * error;
    }
}
