package net.snurkabill.neuralnetworks.benchmark.RBM.Vector;

public class Tuple {
    private static final double TOLERANCE = 0.00001;

    private int[] array;
    private double[] doubleArray;
    private int num;

    public Tuple(int... array) {
        this.array = new int[array.length];
        System.arraycopy(array, 0, this.array, 0, array.length);
        int sum = 0;
        for (int i = 0; i < array.length; i++) {
            sum += Math.pow(2, array.length - (i + 1)) * array[i];
        }
        num = sum;
        this.doubleArray = new double[array.length];
        for (int i = 0; i < doubleArray.length; i++) {
            doubleArray[i] = array[i];
        }
    }

    public Tuple(double[] results) {
        this(convertToInt(results));
    }

    private static final int[] convertToInt(double[] results) {
        int[] array = new int[results.length];
        for (int i = 0; i < array.length; i++) {
            array[i] = (int) (results[i] + TOLERANCE);
        }
        return array;
    }

    public double[] getDoubleArray() {
        return doubleArray;
    }

    public int[] getArray() {
        return array;
    }

    public int getNum() {
        return num;
    }

    public int getLength() {
        return this.array.length;
    }

    @Override
    public String toString() {
        return String.valueOf(num);
    }
}
