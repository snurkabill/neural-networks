package net.snurkabill.neuralnetworks.newdata.database;

import org.junit.Test;

import java.util.Arrays;
import java.util.Enumeration;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class ClassificationDatabaseTest {

    private static final int VECTOR_SIZE = 1;

    private NewDataItem[][] generator(int numOfClasses, int[] numOfElements, int startingPoint) {
        NewDataItem[][] data = new NewDataItem[numOfClasses][];
        for (int i = 0; i < data.length; i++) {
            data[i] = new NewDataItem[numOfElements[i]];
        }
        int helpNum = startingPoint;
        for (int i = 0; i < numOfClasses; i++) {
            for (int j = 0; j < numOfElements[i]; j++) {
                double[] array = new double[VECTOR_SIZE];
                for (int k = 0; k < VECTOR_SIZE; k++) {
                    array[k] = helpNum;
                    helpNum++;
                }
                double[] array2 = new double[numOfClasses];
                for (int k = 0; k < numOfClasses; k++) {
                    array2[k] = helpNum;
                }
                data[i][j] = new NewDataItem(array, array2);
            }
        }
        return data;
    }

    @Test
    public void lenOfClasses() {
        int[] trainSizes = new int[] {1, 2, 3};
        int[] validationSizes = new int[] {4, 5, 6};
        int[] testingSizes = new int[] {7, 8, 9};
        NewDataItem[][] trainingSet = generator(3, trainSizes, 0);
        NewDataItem[][] validationSet = generator(3, validationSizes, 10);
        NewDataItem[][] testingSet = generator(3, testingSizes, 100);

        ClassificationDatabase classificationDatabase = new ClassificationDatabase(trainingSet, validationSet, testingSet,
                null, "name", 0);
        for (int i = 0; i < 3; i++) {
            assertEquals(trainSizes[i], classificationDatabase.getSizeOfTrainSetClass(i));
            assertEquals(validationSizes[i], classificationDatabase.getSizeOfValidationSetClass(i));
            assertEquals(testingSizes[i], classificationDatabase.getSizeOfTestSetClass(i));
        }
    }

    @Test
    public void setSizes() {
        int[] trainSizes = new int[] {1, 2, 3};
        int[] validationSizes = new int[] {4, 5, 6};
        int[] testingSizes = new int[] {7, 8, 9};
        NewDataItem[][] trainingSet = generator(3, trainSizes, 0);
        NewDataItem[][] validationSet = generator(3, validationSizes, 10);
        NewDataItem[][] testingSet = generator(3, testingSizes, 100);

        ClassificationDatabase classificationDatabase = new ClassificationDatabase(trainingSet, validationSet, testingSet,
                null, "name", 0);
        assertEquals(Arrays.stream(trainSizes).sum(), classificationDatabase.getTrainingSetSize());
        assertEquals(Arrays.stream(validationSizes).sum(), classificationDatabase.getValidationSetSize());
        assertEquals(Arrays.stream(testingSizes).sum(), classificationDatabase.getTestingSetSize());
    }

    @Test
    public void hasTarget() {
        int[] trainSizes = new int[] {1, 2, 3};
        int[] validationSizes = new int[] {4, 5, 6};
        int[] testingSizes = new int[] {7, 8, 9};
        NewDataItem[][] trainingSet = generator(3, trainSizes, 0);
        NewDataItem[][] validationSet = generator(3, validationSizes, 10);
        NewDataItem[][] testingSet = generator(3, testingSizes, 100);

        ClassificationDatabase classificationDatabase = new ClassificationDatabase(trainingSet, validationSet, testingSet,
                null, "name", 0);
        assertTrue(classificationDatabase.hasTarget());
    }

    @Test
    public void targetSize() {
        int[] trainSizes = new int[] {1, 2, 3};
        int[] validationSizes = new int[] {4, 5, 6};
        int[] testingSizes = new int[] {7, 8, 9};
        NewDataItem[][] trainingSet = generator(3, trainSizes, 0);
        NewDataItem[][] validationSet = generator(3, validationSizes, 10);
        NewDataItem[][] testingSet = generator(3, testingSizes, 100);

        ClassificationDatabase classificationDatabase = new ClassificationDatabase(trainingSet, validationSet, testingSet,
                null, "name", 0);
        assertEquals(3, classificationDatabase.getTargetSize());
    }

    @Test
    public void vectorSize() {
        int[] trainSizes = new int[] {1, 2, 3};
        int[] validationSizes = new int[] {4, 5, 6};
        int[] testingSizes = new int[] {7, 8, 9};
        NewDataItem[][] trainingSet = generator(3, trainSizes, 0);
        NewDataItem[][] validationSet = generator(3, validationSizes, 10);
        NewDataItem[][] testingSet = generator(3, testingSizes, 100);
        ClassificationDatabase classificationDatabase = new ClassificationDatabase(trainingSet, validationSet, testingSet,
                null, "name", 0);
        assertEquals(VECTOR_SIZE, classificationDatabase.getVectorSize());
    }

    @Test
    public void validationSetEnumerator() {
        int[] trainSizes = new int[] {1, 2, 3};
        int[] validationSizes = new int[] {4, 5, 6};
        int[] testingSizes = new int[] {7, 8, 9};
        NewDataItem[][] trainingSet = generator(3, trainSizes, 0);
        NewDataItem[][] validationSet = generator(3, validationSizes, 10);
        NewDataItem[][] testingSet = generator(3, testingSizes, 100);
        ClassificationDatabase classificationDatabase = new ClassificationDatabase(trainingSet, validationSet, testingSet,
                null, "name", 0);
        Enumeration<NewDataItem> enumerator = classificationDatabase.createValidationSetEnumerator();

        int counter = 10; // starting element
        for (int i = 0; i < validationSizes.length; i++) {
            for (int j = 0; j < validationSizes[i]; j++) {
                NewDataItem item = enumerator.nextElement();
                assertEquals(counter, item.data[0], 0.000_000_1);
                counter++;
            }
        }
    }

    @Test
    public void testingSetEnumerator() {
        int[] trainSizes = new int[] {1, 2, 3};
        int[] validationSizes = new int[] {4, 5, 6};
        int[] testingSizes = new int[] {7, 8, 9};
        NewDataItem[][] trainingSet = generator(3, trainSizes, 0);
        NewDataItem[][] validationSet = generator(3, validationSizes, 10);
        NewDataItem[][] testingSet = generator(3, testingSizes, 100);
        ClassificationDatabase classificationDatabase = new ClassificationDatabase(trainingSet, validationSet, testingSet,
                null, "name", 0);
        Enumeration<NewDataItem> enumerator = classificationDatabase.createTestingSetEnumerator();

        int counter = 100; // starting element
        for (int i = 0; i < testingSizes.length; i++) {
            for (int j = 0; j < testingSizes[i]; j++) {
                NewDataItem item = enumerator.nextElement();
                assertEquals(counter, item.data[0], 0.000_000_1);
                counter++;
            }
        }
    }

    @Test
    public void testingSetEnumeratorOverClass() {
        int[] trainSizes = new int[] {1, 2, 3};
        int[] validationSizes = new int[] {4, 5, 6};
        int[] testingSizes = new int[] {7, 8, 9};
        NewDataItem[][] trainingSet = generator(3, trainSizes, 0);
        NewDataItem[][] validationSet = generator(3, validationSizes, 10);
        NewDataItem[][] testingSet = generator(3, testingSizes, 100);
        ClassificationDatabase classificationDatabase = new ClassificationDatabase(trainingSet, validationSet, testingSet,
                null, "name", 0);

        int counter = 100; // starting element
        for (int k = 0; k < testingSizes.length; k++) {
            Enumeration<NewDataItem> enumerator = classificationDatabase.createTestClassEnumerator(k);
            int i = 0;
            for (; enumerator.hasMoreElements(); i++) {
                NewDataItem item = enumerator.nextElement();
                assertEquals(counter, item.data[0], 0.000_000_1);
                counter++;
            }
            assertEquals(testingSizes[k], i);
        }
    }

    @Test
    public void validationSetEnumeratorOverClass() {
        int[] trainSizes = new int[] {1, 2, 3};
        int[] validationSizes = new int[] {4, 5, 6};
        int[] testingSizes = new int[] {7, 8, 9};
        NewDataItem[][] trainingSet = generator(3, trainSizes, 0);
        NewDataItem[][] validationSet = generator(3, validationSizes, 10);
        NewDataItem[][] testingSet = generator(3, testingSizes, 100);
        ClassificationDatabase classificationDatabase = new ClassificationDatabase(trainingSet, validationSet, testingSet,
                null, "name", 0);

        int counter = 10; // starting element
        for (int k = 0; k < testingSizes.length; k++) {
            Enumeration<NewDataItem> enumerator = classificationDatabase.createValidationClassEnumerator(k);
            int i = 0;
            for (; enumerator.hasMoreElements(); i++) {
                NewDataItem item = enumerator.nextElement();
                assertEquals(counter, item.data[0], 0.000_000_1);
                counter++;
            }
            assertEquals(validationSizes[k], i);
        }
    }

    @Test
    public void trainingSetEnumeratorOverClass() {
        int[] trainSizes = new int[] {1, 2, 3};
        int[] validationSizes = new int[] {4, 5, 6};
        int[] testingSizes = new int[] {7, 8, 9};
        NewDataItem[][] trainingSet = generator(3, trainSizes, 0);
        NewDataItem[][] validationSet = generator(3, validationSizes, 10);
        NewDataItem[][] testingSet = generator(3, testingSizes, 100);
        ClassificationDatabase classificationDatabase = new ClassificationDatabase(trainingSet, validationSet, testingSet,
                null, "name", 0);

        int counter = 0; // starting element
        for (int k = 0; k < testingSizes.length; k++) {
            Enumeration<NewDataItem> enumerator = classificationDatabase.createTrainingClassEnumerator(k);
            int i = 0;
            for (; enumerator.hasMoreElements(); i++) {
                NewDataItem item = enumerator.nextElement();
                assertEquals(counter, item.data[0], 0.000_000_1);
                counter++;
            }
            assertEquals(trainSizes[k], i);
        }
    }

    @Test
    public void trainingInfiniteSimpleSetEnumerator() {
        int[] trainSizes = new int[] {1, 2, 3};
        int[] validationSizes = new int[] {4, 5, 6};
        int[] testingSizes = new int[] {7, 8, 9};
        NewDataItem[][] trainingSet = generator(3, trainSizes, 0);
        NewDataItem[][] validationSet = generator(3, validationSizes, 10);
        NewDataItem[][] testingSet = generator(3, testingSizes, 100);

        ClassificationDatabase classificationDatabase = new ClassificationDatabase(trainingSet, validationSet, testingSet,
                null, "name", 0);
        Enumeration<NewDataItem> enumerator = classificationDatabase.createInfiniteSimpleTrainingSetEnumerator();
        NewDataItem item = enumerator.nextElement();
        assertEquals(0, item.data[0], 0.000_000_1);
        item = enumerator.nextElement();
        assertEquals(1, item.data[0], 0.000_000_1);
        item = enumerator.nextElement();
        assertEquals(3, item.data[0], 0.000_000_1);
        item = enumerator.nextElement();
        assertEquals(0, item.data[0], 0.000_000_1);
        item = enumerator.nextElement();
        assertEquals(2, item.data[0], 0.000_000_1);
        item = enumerator.nextElement();
        assertEquals(4, item.data[0], 0.000_000_1);
        item = enumerator.nextElement();
        assertEquals(0, item.data[0], 0.000_000_1);
        item = enumerator.nextElement();
        assertEquals(1, item.data[0], 0.000_000_1);
        item = enumerator.nextElement();
        assertEquals(5, item.data[0], 0.000_000_1);
    }

    @Test
    public void trainingInfiniteRandomSetEnumerator() {
        int[] trainSizes = new int[] {1, 2, 3};
        int[] validationSizes = new int[] {4, 5, 6};
        int[] testingSizes = new int[] {7, 8, 9};
        NewDataItem[][] trainingSet = generator(3, trainSizes, 0);
        NewDataItem[][] validationSet = generator(3, validationSizes, 10);
        NewDataItem[][] testingSet = generator(3, testingSizes, 100);

        ClassificationDatabase classificationDatabase = new ClassificationDatabase(trainingSet, validationSet, testingSet,
                null, "name", 0);
        Enumeration<NewDataItem> enumerator = classificationDatabase.createInfiniteRandomTrainingSetEnumerator();
        for (int i = 0; i < 1000; i++) {
            NewDataItem item = enumerator.nextElement();
        }
    }

}
