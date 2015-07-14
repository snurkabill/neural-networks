package net.snurkabill.neuralnetworks.math.linear.normalization;

public class UnitBasedNormalizer extends GeneralNormalizer {

    public static UnitBasedNormalizer buildUnitBasedNormalizer(double[][] data, double lowLimit, double topLimit) {
        if(data == null) {
            throw new IllegalArgumentException("Array can't be null");
        }
        double[] max_ = new double[data[0].length];
        double[] min_ = new double[data[0].length];
        for (int i = 0; i < data[0].length; i++) {
            double max = Double.NEGATIVE_INFINITY;
            double min = Double.POSITIVE_INFINITY;
            for (int j = 0; j < data.length; j++) {
                double item = data[j][i];
                if (item > max) {
                    max = item;
                }
                if (item < min) {
                    min = item;
                }
            }
            max_[i] = max;
            min_[i] = min;
        }
        return new UnitBasedNormalizer(data[0].length, max_, min_, lowLimit, topLimit);
    }

    private double[] max;
    private double[] min;

    private double topLimit;
    private double lowLimit;

    public UnitBasedNormalizer(int vectorSize, double[] max, double[] min, double lowLimit, double topLimit) {
        super(vectorSize);
        this.max = max;
        this.min = min;
        this.lowLimit = lowLimit;
        this.topLimit = topLimit;
    }

    @Override
    public double[] transform(double[] vector) {
        checkSize(vector);
        double[] array = new double[vectorSize];
        for (int i = 0; i < vectorSize; i++) {
            array[i] = lowLimit + ((vector[i] - min[i]) * (topLimit - lowLimit) / (max[i] - min[i]));
        }
        return array;
    }
}
