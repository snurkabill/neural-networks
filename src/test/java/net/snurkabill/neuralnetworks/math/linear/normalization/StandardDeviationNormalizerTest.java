package net.snurkabill.neuralnetworks.math.linear.normalization;

import net.snurkabill.neuralnetworks.utilities.Utilities;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class StandardDeviationNormalizerTest {

    @Test
    public void simpleTest() {
        double[] data = new double[8];
        data[0] = 2;
        data[1] = 4;
        data[2] = 4;
        data[3] = 4;
        data[4] = 5;
        data[5] = 5;
        data[6] = 7;
        data[7] = 9;
        assertEquals(2, Utilities.stddev(data), 0.000_000_1);
    }

    @Test
    public void stdev() {
        double[][] data = new double[8][1];
        data[0][0] = 2;
        data[1][0] = 4;
        data[2][0] = 4;
        data[3][0] = 4;
        data[4][0] = 5;
        data[5][0] = 5;
        data[6][0] = 7;
        data[7][0] = 9;

        StandardDeviationNormalizer normalizer = StandardDeviationNormalizer.buildStandardDeviationNormalizer(data);
        assertArrayEquals(new double[]{0.0}, normalizer.transform(data[4]), 0.000_000_1);
        assertArrayEquals(new double[]{0.0}, normalizer.transform(data[5]), 0.000_000_1);
        assertArrayEquals(new double[]{1.0}, normalizer.transform(data[6]), 0.000_000_1);
    }
}
