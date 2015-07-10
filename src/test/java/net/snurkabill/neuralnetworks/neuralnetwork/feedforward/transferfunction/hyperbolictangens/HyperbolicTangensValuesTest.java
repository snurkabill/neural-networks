package net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction.hyperbolictangens;

import net.snurkabill.neuralnetworks.Tolerance;
import net.snurkabill.neuralnetworks.math.function.transferfunction.HyperbolicTangens;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;

import static org.junit.Assert.assertEquals;

@RunWith(Parameterized.class)
public class HyperbolicTangensValuesTest {

    private HyperbolicTangens function;
    private double expectedResults;
    private double input;

    @Before
    public void initialize() {
        function = new HyperbolicTangens();
    }

    public HyperbolicTangensValuesTest(double input, double expectedResults) {
        this.expectedResults = expectedResults;
        this.input = input;
    }

    @Parameterized.Parameters
    public static Collection inputValue() {
        return Arrays.asList(new Object[][]{
                {0.0, 0.0},
                {100, HyperbolicTangens.TOP_LIMIT},
                {-100, HyperbolicTangens.BOTTOM_LIMIT},
                {0.25, 0.2449186624},
                {-0.25, -0.2449186624},
                {0.5, 0.4621171572},
                {-0.5, -0.4621171572},
                {1.0, 0.7615941559},
                {-1.0, -0.7615941559},
                {2.0, 0.9640275800},
                {-2.0, -0.9640275800},
                {3.0, 0.9950547536},
                {-3.0, -0.9950547536},
        });
    }

    @Test
    public void testFunction() {
        assertEquals(expectedResults, function.calculateOutputValue(input),
                Tolerance.TOLERANCE);
    }
}
