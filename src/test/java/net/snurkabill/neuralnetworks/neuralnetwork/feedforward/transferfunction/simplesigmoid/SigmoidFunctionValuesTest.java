package net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction.simplesigmoid;

import net.snurkabill.neuralnetworks.Tolerance;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction.SigmoidFunction;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;

import static org.junit.Assert.assertEquals;

@RunWith(Parameterized.class)
public class SigmoidFunctionValuesTest {

    private SigmoidFunction function;
    private double expectedResults;
    private double input;

    @Before
    public void initialize() {
        function = new SigmoidFunction();
    }

    public SigmoidFunctionValuesTest(double input, double expectedResults) {
        this.expectedResults = expectedResults;
        this.input = input;
    }

    @Parameterized.Parameters
    public static Collection inputValue() {
        return Arrays.asList(new Object[][]{
                {0.0, 0.5},
                {1.0, 0.7310585786},
                {-1.0, 0.2689414213},
                {2.0, 0.8807970779},
                {-2.0, 0.1192029220},
                {3.0, 0.9525741268},
                {-3.0, 0.0474258731},
                {100, SigmoidFunction.TOP_LIMIT},
                {-100, SigmoidFunction.BOTTOM_LIMIT},
        });
    }

    @Test
    public void testFunction() {
        assertEquals(expectedResults, function.calculateOutputValue(input),
                Tolerance.TOLERANCE);
    }
}
