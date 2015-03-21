package net.snurkabill.neuralnetworks.utilities;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class UtilitiesTest {

    private static double tolerance = 1 * Math.pow(10, -6);

    @Test(expected=IllegalArgumentException.class)
    public void meanEmptyArray() {
        assertEquals(0.0, Utilities.mean(new double[] {}), tolerance);
    }

    @Test
    public void mean() {
        assertEquals(5.5, Utilities.mean(new double [] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}), tolerance);
        assertEquals(5.5, Utilities.mean(new double [] {1, 10}), tolerance);
        assertEquals(0.0, Utilities.mean(new double [] {-5, 5}), tolerance);
        assertEquals(-5.0, Utilities.mean(new double [] {-10, 0}), tolerance);
        assertEquals(0.0, Utilities.mean(new double [] {-1000, 1000, 0, 1, -1, -42, 42}), tolerance);
        assertEquals(0.0, Utilities.mean(new double [] {0}), tolerance);
        assertEquals(1.0, Utilities.mean(new double [] {1}), tolerance);
        assertEquals(10000.0, Utilities.mean(new double [] {10000}), tolerance);
        assertEquals(-10.0, Utilities.mean(new double [] {-10}), tolerance);
        assertEquals(5.0, Utilities.mean(new double [] {4, 4, 4, 2, 5, 5, 7, 9}), tolerance);
    }

    @Test(expected=IllegalArgumentException.class)
    public void stdevEmtpyArray() {
        Utilities.stddev(new double[] {}, 0);
    }

    @Test(expected=IllegalArgumentException.class)
    public void stdevOneElementArray() {
        Utilities.stddev(new double[] {1}, 0);
    }

    @Test
    public void stddev() {
        assertEquals(0.0, Utilities.stddev(new double[] {1, 1, 1, 1, 1}), tolerance);
        assertEquals(0.0, Utilities.stddev(new double[] {2, 2, 2, 2, 2}), tolerance);
        assertEquals(0.0, Utilities.stddev(new double[] {-1, -1, -1, -1, -1}), tolerance);
        assertEquals(2.1380899352, Utilities.stddev(new double [] {4, 4, 4, 2, 5, 5, 7, 9}), tolerance);
        assertEquals(3.0607876523, Utilities.stddev(
                new double[] {9, 2, 5, 4, 12, 7, 8, 11, 9, 3, 7, 4, 12, 5, 4, 10, 9, 6, 9, 4}), tolerance);
        double[] array = new double[1_000_000];
        for (int i = 0; i < 1_000_000; i++) {
            array[i] = i % 2 == 0 ? 0 : -2;
        }
        assertEquals(1, Utilities.stddev(array), tolerance);
    }

}
