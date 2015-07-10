package net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction.parametrizedsigmoid;

import net.snurkabill.neuralnetworks.Tolerance;
import net.snurkabill.neuralnetworks.math.function.transferfunction.ParametrizedSigmoidFunctionForBipolarVector;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;

import static org.junit.Assert.assertEquals;

@RunWith(Parameterized.class)
public class ParametrizedSigmoidFunctionValuesTest {

    private ParametrizedSigmoidFunctionForBipolarVector function;
    private double expectedResults;
    private double input;

    @Before
    public void initialize() {
        function = new ParametrizedSigmoidFunctionForBipolarVector();
    }

    public ParametrizedSigmoidFunctionValuesTest(double input, double expectedResults) {
        this.expectedResults = expectedResults;
        this.input = input;
    }

    @Parameterized.Parameters
    public static Collection inputValue() {
        return Arrays.asList(new Object[][]{
                {0.0, 0.0},
                {1.0, 0.2310585786},
                {-1.0, -0.2310585786},
                {2.0, 0.3807970779},
                {-2.0, -0.3807970779},
                {3.0, 0.4525741268},
                {-3.0, -0.4525741268},
                {100, ParametrizedSigmoidFunctionForBipolarVector.TOP_LIMIT},
                {-100, ParametrizedSigmoidFunctionForBipolarVector.BOTTOM_LIMIT},
        });
    }

    @Test
    public void testFunction() {
        assertEquals(expectedResults, function.calculateOutputValue(input),
                Tolerance.TOLERANCE);
    }
}
