package net.snurkabill.neuralnetworks.newdata.database;

import org.junit.Test;

import java.util.Enumeration;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

public class RegressionDatabaseTest {

    private static final int TESTING_VECTOR_SIZE = 3;

    private NewDataItem[] generator(int numOfElements, int startingPoint) {
        NewDataItem[] data = new NewDataItem[numOfElements];
        int helpNum = startingPoint;
        for (int i = 0; i < numOfElements; i++) {
            double[] array = new double[TESTING_VECTOR_SIZE];
            for (int j = 0; j < TESTING_VECTOR_SIZE; j++) {
                array[j] = helpNum;
                helpNum++;
            }
            data[i] = new NewDataItem(array);
        }
        return data;
    }

    @Test
    public void setsSizes() {
        NewDataItem[] trainingSet = generator(1, 0);
        NewDataItem[] validationSet = generator(2, 10);
        NewDataItem[] testingSet = generator(3, 100);

        RegressionDatabase regressionDatabase = new RegressionDatabase(trainingSet, validationSet, testingSet,
                "name", 0);
        assertEquals(trainingSet.length, regressionDatabase.getTrainingSetSize());
        assertEquals(validationSet.length, regressionDatabase.getValidationSetSize());
        assertEquals(testingSet.length, regressionDatabase.getTestingSetSize());
    }

    @Test
    public void hasTarget() {
        NewDataItem[] trainingSet = generator(1, 0);
        NewDataItem[] validationSet = generator(2, 10);
        NewDataItem[] testingSet = generator(3, 100);

        RegressionDatabase regressionDatabase = new RegressionDatabase(trainingSet, validationSet, testingSet,
                "name", 0);
        assertFalse(regressionDatabase.hasTarget());
    }

    @Test
    public void targetSize() {
        NewDataItem[] trainingSet = new NewDataItem[1];
        trainingSet[0] = new NewDataItem(new double[1], new double[2]);

        RegressionDatabase regressionDatabase = new RegressionDatabase(trainingSet, trainingSet, trainingSet,
                "name", 0);
        assertEquals(2, regressionDatabase.getTargetSize());
    }

    @Test
    public void vectorSize() {
        NewDataItem[] trainingSet = new NewDataItem[1];
        trainingSet[0] = new NewDataItem(new double[1], new double[2]);

        RegressionDatabase regressionDatabase = new RegressionDatabase(trainingSet, trainingSet, trainingSet,
                "name", 0);
        assertEquals(1, regressionDatabase.getVectorSize());
    }

    @Test
    public void testingSetEnumerator() {
        NewDataItem[] trainingSet = generator(1, 0);
        NewDataItem[] validationSet = generator(2, 10);
        NewDataItem[] testingSet = generator(3, 100);

        RegressionDatabase regressionDatabase = new RegressionDatabase(trainingSet, validationSet, testingSet,
                "name", 0);

        Enumeration<NewDataItem> enumerator = regressionDatabase.createTestingSetEnumerator();
        int count = 0;
        for (int i = 100; enumerator.hasMoreElements(); i++, count++) {
            NewDataItem item = enumerator.nextElement();
            for (int j = 0; j < item.getDataVectorSize(); j++, i++) {
                assertEquals(i, item.data[j], 0.000_000_1);
            }
            i--;
        }
        assertEquals(3, count);
    }

    @Test
    public void validationSetEnumerator() {
        NewDataItem[] trainingSet = generator(1, 0);
        NewDataItem[] validationSet = generator(2, 10);
        NewDataItem[] testingSet = generator(3, 100);

        RegressionDatabase regressionDatabase = new RegressionDatabase(trainingSet, validationSet, testingSet,
                "name", 0);

        Enumeration<NewDataItem> enumerator = regressionDatabase.createValidationSetEnumerator();
        int count = 0;
        for (int i = 10; enumerator.hasMoreElements(); i++, count++) {
            NewDataItem item = enumerator.nextElement();
            for (int j = 0; j < item.getDataVectorSize(); j++, i++) {
                assertEquals(i, item.data[j], 0.000_000_1);
            }
            i--;
        }
        assertEquals(2, count);
    }

    @Test
    public void trainingSetEnumerator() {
        NewDataItem[] trainingSet = generator(5, 0);
        NewDataItem[] validationSet = generator(2, 10);
        NewDataItem[] testingSet = generator(3, 100);

        RegressionDatabase regressionDatabase = new RegressionDatabase(trainingSet, validationSet, testingSet,
                "name", 0);

        Enumeration<NewDataItem> enumerator = regressionDatabase.createTrainingSetEnumerator();
        int count = 0;
        for (int i = 0; enumerator.hasMoreElements(); i++, count++) {
            NewDataItem item = enumerator.nextElement();
            for (int j = 0; j < item.getDataVectorSize(); j++, i++) {
                assertEquals(i, item.data[j], 0.000_000_1);
            }
            i--;
        }
        assertEquals(5, count);
    }

    @Test
    public void trainingInfiniteSimpleSetEnumerator() {
        NewDataItem[] trainingSet = generator(5, 0);
        NewDataItem[] validationSet = generator(2, 10);
        NewDataItem[] testingSet = generator(3, 100);

        RegressionDatabase regressionDatabase = new RegressionDatabase(trainingSet, validationSet, testingSet,
                "name", 0);

        Enumeration<NewDataItem> enumerator = regressionDatabase.createInfiniteSimpleTrainingSetEnumerator();
        int count = 0;

        for (int i = 0; i < 5 * TESTING_VECTOR_SIZE; i++, count++) {
            NewDataItem item = enumerator.nextElement();
            for (int j = 0; j < item.getDataVectorSize(); j++, i++) {
                assertEquals(i, item.data[j], 0.000_000_1);
            }
            i--;
        }
        for (int i = 0; i < 5 * TESTING_VECTOR_SIZE; i++, count++) {
            NewDataItem item = enumerator.nextElement();
            for (int j = 0; j < item.getDataVectorSize(); j++, i++) {
                assertEquals(i, item.data[j], 0.000_000_1);
            }
            i--;
        }
    }

    @Test
    public void trainingInfiniteRandomSetEnumerator() {
        NewDataItem[] trainingSet = generator(5, 0);
        NewDataItem[] validationSet = generator(2, 10);
        NewDataItem[] testingSet = generator(3, 100);

        RegressionDatabase regressionDatabase = new RegressionDatabase(trainingSet, validationSet, testingSet,
                "name", 0);

        Enumeration<NewDataItem> enumerator = regressionDatabase.createInfiniteRandomTrainingSetEnumerator();
        for (int i = 0; i < 1000; i++) {
            NewDataItem item = enumerator.nextElement();
        }
    }

}
