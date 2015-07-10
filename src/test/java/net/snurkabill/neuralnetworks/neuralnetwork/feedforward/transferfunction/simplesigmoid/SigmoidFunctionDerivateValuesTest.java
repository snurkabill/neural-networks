package net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction.simplesigmoid;

import net.snurkabill.neuralnetworks.Tolerance;
import net.snurkabill.neuralnetworks.math.function.transferfunction.ParametrizedSigmoidFunctionForBipolarVector;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;

import static junit.framework.Assert.assertEquals;

@RunWith(Parameterized.class)
public class SigmoidFunctionDerivateValuesTest {

    private ParametrizedSigmoidFunctionForBipolarVector function;
    private double expectedResults;
    private double input;

    @Before
    public void initialize() {
        function = new ParametrizedSigmoidFunctionForBipolarVector();
    }

    public SigmoidFunctionDerivateValuesTest(double input, double expectedResults) {
        this.expectedResults = expectedResults;
        this.input = input;
    }

    @Parameterized.Parameters
    public static Collection inputValue() {
        return Arrays.asList(new Object[][]{
                {0.0, 0.25},
                {0.25, 0.2461340827},
                {-0.25, 0.2461340827},
                {0.5, 0.235003712},
                {-0.5, 0.235003712},
                {1.0, 0.1966119332},
                {-1.0, 0.1966119332},
                {2.0, 0.104993585},
                {-2.0, 0.104993585},
                {3.0, 0.0451766597},
                {-3.0, 0.0451766597},
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
