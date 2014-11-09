package net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction.softplus;

import net.snurkabill.neuralnetworks.Tolerance;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction.SoftPlus;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;

import static junit.framework.TestCase.assertEquals;

@RunWith(Parameterized.class)
public class SoftPlusFunctionValuesTest {

    private SoftPlus function;
    private double expectedResults;
    private double input;

    @Before
    public void initialize() {
        function = new SoftPlus();
    }

    public SoftPlusFunctionValuesTest(double input, double expectedResults) {
        this.expectedResults = expectedResults;
        this.input = input;
    }

    @Parameterized.Parameters
    public static Collection inputValue() {
        return Arrays.asList(new Object[][]{
                {0.0, 0.693147180},
                {1.0, 1.3132616875},
                {-1.0, 0.3132616875},
                {2.0, 2.1269280110},
                {-2.0, 0.1269280110},
                {3.0, 3.0485873515},
                {-3.0, 0.0485873515},
                {100, 100},
                {-100, 0},
        });
    }

    @Test
    public void testFunction() {
        assertEquals(expectedResults, function.calculateOutputValue(input),
                Tolerance.TOLERANCE);
    }
}
