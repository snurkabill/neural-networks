package net.snurkabill.neuralnetworks.utilities;

public class Utilities {
	
	public static double mean(final double[] input) {
        double m = 0.0;
        for (int i=0; i<input.length; i++)
            m += input[i];

        return m/input.length;
    }
	
	public static double stddev(final double[] input, double mean) {
        double sum = 0.0;
        for (int i=0; i<input.length; i++)
            sum += (input[i] - mean) * (input[i] - mean);

        return Math.sqrt(sum/(input.length - 1));
    }

	public static double sigmoid(double x) {
        return (1.0 / (1.0 + Math.exp(-x)));
    }
	
	public static double sigmoidDerivative(double x) {
		double tmp = Math.exp(x);
		return tmp / ((tmp + 1.0) * (tmp + 1.0));
	}

    public static double secondsSpent(long from, long to) {
        return ((to - from) / 1000.0);
    }
}
