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
public class HyperbolicTangensDerivateValuesTest {

    private HyperbolicTangens function;
    private double expectedResults;
    private double input;

    @Before
    public void initialize() {
        function = new HyperbolicTangens();
    }

    public HyperbolicTangensDerivateValuesTest(double input, double expectedResults) {
        this.expectedResults = expectedResults;
        this.input = input;
    }

    @Parameterized.Parameters
    public static Collection inputValue() {
        return Arrays.asList(new Object[][]{
                {0.0, 1.0},
                {0.25, 0.9400148488},
                {-0.25, 0.9400148488},
                {0.5, 0.7864477329},
                {-0.5, 0.7864477329},
                {1.0, 0.4199743416},
                {-1.0, 0.4199743416},
                {2.0, 0.0706508248},
                {-2.0, 0.0706508248},
                {3.0, 0.0098660371},
                {-3.0, 0.0098660371},
                {100, 0.0},
                {-100, 0.0},
        });
    }

    @Test
    public void testFunction() {
        assertEquals("Input: " + String.valueOf(input), expectedResults, function.calculateDerivative(input),
                Tolerance.TOLERANCE);
    }
}
