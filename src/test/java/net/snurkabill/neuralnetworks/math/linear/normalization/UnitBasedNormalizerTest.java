package net.snurkabill.neuralnetworks.math.linear.normalization;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public class UnitBasedNormalizerTest {

    @Test
    public void simpleTest() {
        double[] first = new double[] {1, -10};
        double[] second = new double[] {10, -1};
        double[] third = new double[] {10, -1};
        double[] fourth = new double[] {10, -1};
        double[] fifth = new double[] {10, -1};
        double[] sixth = new double[] {10, -1};

        double[][] array = new double[6][];

        array[0] = first;
        array[1] = second;
        array[2] = third;
        array[3] = fourth;
        array[4] = fifth;
        array[5] = sixth;

        UnitBasedNormalizer normalizer = UnitBasedNormalizer.buildUnitBasedNormalizer(array, 0, 1);

        assertArrayEquals(new double[]{0, 0}, normalizer.transform(first), 0.000_000_1);
        assertArrayEquals(new double[]{1, 1}, normalizer.transform(second), 0.000_000_1);
        assertArrayEquals(new double[]{1, 1}, normalizer.transform(third), 0.000_000_1);
        assertArrayEquals(new double[]{1, 1}, normalizer.transform(fourth), 0.000_000_1);
        assertArrayEquals(new double[]{1, 1}, normalizer.transform(fifth), 0.000_000_1);
        assertArrayEquals(new double[]{1, 1}, normalizer.transform(sixth), 0.000_000_1);
    }

    @Test
    public void shiftedTest() {
        double[] first = new double[] {1, 10};
        double[] second = new double[] {2, -1};
        double[] third = new double[] {3, -1};
        double[] fourth = new double[] {4, -10};
        double[] fifth = new double[] {5, 0};

        double[][] array = new double[5][];

        array[0] = first;
        array[1] = second;
        array[2] = third;
        array[3] = fourth;
        array[4] = fifth;

        UnitBasedNormalizer normalizer = UnitBasedNormalizer.buildUnitBasedNormalizer(array, -2, -1);

        assertArrayEquals(new double[]{-2, -1}, normalizer.transform(first), 0.000_000_1);
        assertArrayEquals(new double[]{-1.75, -1.55}, normalizer.transform(second), 0.000_000_1);
        assertArrayEquals(new double[]{-1.5, -1.55}, normalizer.transform(third), 0.000_000_1);
        assertArrayEquals(new double[]{-1.25, -2}, normalizer.transform(fourth), 0.000_000_1);
        assertArrayEquals(new double[]{-1, -1.5}, normalizer.transform(fifth), 0.000_000_1);
    }
}
