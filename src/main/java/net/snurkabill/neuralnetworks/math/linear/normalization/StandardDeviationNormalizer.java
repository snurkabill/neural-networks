package net.snurkabill.neuralnetworks.math.linear.normalization;

import net.snurkabill.neuralnetworks.utilities.Utilities;

public class StandardDeviationNormalizer extends GeneralNormalizer {

    public static StandardDeviationNormalizer buildStandardDeviationNormalizer(double[][] data) {
        if(data == null) {
            throw new IllegalArgumentException("Array can't be null");
        }
        double[] means = new double[data[0].length];
        double[] stdev = new double[data[0].length];
        for (int i = 0; i < data[0].length; i++) {
            double[] array = new double[data.length];
            for (int j = 0; j < data.length; j++) {
                array[j] = data[j][i];
            }
            means[i] = Utilities.mean(array);
            stdev[i] = Utilities.stddev(array, means[i]);
        }
        return new StandardDeviationNormalizer(data[0].length, means, stdev);
    }

    private double[] means;
    private double[] stdev;

    public StandardDeviationNormalizer(int vectorSize, double[] means, double[] stdev) {
        super(vectorSize);
        this.means = means;
        this.stdev = stdev;
    }

    @Override
    public double[] transform(double[] vector) {
        checkSize(vector);
        double[] array = new double[vectorSize];
        for (int i = 0; i < vectorSize; i++) {
            if(stdev[i] == 0) {
                array[i] = 0.0;
            } else {
                array[i] = (vector[i] - means[i]) / stdev[i];
            }
        }
        return array;
    }
}
