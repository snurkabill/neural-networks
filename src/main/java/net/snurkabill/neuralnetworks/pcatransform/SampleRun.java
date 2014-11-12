package net.snurkabill.neuralnetworks.pcatransform;

import Jama.Matrix;

/**
 * An example program using the library
 */
public class SampleRun {
    public static void main(String[] args) {
        System.out.println("Running a demonstrational program on some sample data ...");
        Matrix trainingData = new Matrix(new double[][]{
                {1, 2, 3, 4, 5, 6},
                {6, 5, 4, 3, 2, 1},
                {2, 2, 2, 2, 2, 2}});
        PCA pca = new PCA(trainingData);
        Matrix testData = new Matrix(new double[][]{
                {1, 2, 3, 4, 5, 6},
                {1, 2, 1, 2, 1, 2}});
        Matrix transformedData =
                pca.transform(testData, PCA.TransformationType.WHITENING);
        System.out.println("Transformed data:");
        for (int r = 0; r < transformedData.getRowDimension(); r++) {
            for (int c = 0; c < transformedData.getColumnDimension(); c++) {
                System.out.print(transformedData.get(r, c));
                if (c == transformedData.getColumnDimension() - 1) continue;
                System.out.print(", ");
            }
            System.out.println("");
        }
    }
}