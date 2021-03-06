package net.snurkabill.neuralnetworks.utilities;

import Jama.Matrix;
import net.snurkabill.neuralnetworks.data.database.DataItem;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Utilities {

    public static final double ERROR_MODIFIER = 0.5;

    public static double mean(final double[] input) {
        if(input.length == 0) {
            throw new IllegalArgumentException("There must be at least one element for calculating mean");
        }
        double m = 0.0;
        for (double anInput : input) {
            m += anInput;
        }
        return m / input.length;
    }

    public static double variance(final double[] input, double mean) {
        if(input.length == 0) {
            throw new IllegalArgumentException("There needs to be at least 1 element for calculating " +
                    "standard deviation");
        }
        double sum = 0.0;
        for (double anInput : input) {
            sum += (anInput - mean) * (anInput - mean);
        }
        return sum / input.length;
    }

    public static double variance(final double[] input) {
        return variance(input, mean(input));
    }

    public static double stddev(final double[] input) {
        return stddev(input, mean(input));
    }

    public static double stddev(final double[] input, double mean) {
        return Math.sqrt(variance(input, mean));
    }

    public static double sigmoid(double x) {
        return (1.0 / (1.0 + Math.exp(-x)));
    }

    public static double sigmoidDerivative(double x) {
        double tmp = Math.exp(x);
        return tmp / ((tmp + 1.0) * (tmp + 1.0));
    }

    public static double calcError(double[] expectedValues, double[] realValues) {
        if (expectedValues.length != realValues.length) {
            throw new IllegalArgumentException("Lengths of arrays are different!");
        }
        double error = 0.0;
        for (int i = 0; i < expectedValues.length; i++) {
            double diff = expectedValues[i] - realValues[i];
            error += diff * diff;
        }
        return error;
    }

    public static Matrix buildMatrix(Map<Integer, List<DataItem>> data) {
        int totalNumOfVectors = 0;
        for (int i = 0; i < data.size(); i++) {
            totalNumOfVectors += data.get(i).size();
        }
        System.out.println("Creating matrix for: " + totalNumOfVectors + " x " + data.get(0).get(0).data.length);
        double[][] rawData = new double[totalNumOfVectors][];
        int lastUsedIndex = 0;
        for (int i = 0; i < data.size(); i++) {
            for (int j = 0; j < data.get(i).size(); j++) {
                rawData[lastUsedIndex] = data.get(i).get(j).data;
                lastUsedIndex++;
            }
        }
        return new Matrix(rawData, totalNumOfVectors, data.get(0).get(0).data.length);
    }

    public static Matrix buildMatrix(Map<Integer, List<DataItem>> data, int numOfElements) {
        Map<Integer, List<DataItem>> newMap = new HashMap<>(data.size());
        for (int i = 0; i < data.size(); i++) {
            newMap.put(i, new ArrayList<DataItem>());
        }
        for (int i = 0; i < numOfElements; i++) {
            newMap.get(i % data.size()).add(data.get(i % data.size()).get(i / data.size()));
        }
        return buildMatrix(newMap);
    }
}
