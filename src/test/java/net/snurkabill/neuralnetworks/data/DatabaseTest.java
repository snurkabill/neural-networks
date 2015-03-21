package net.snurkabill.neuralnetworks.data;

import net.snurkabill.neuralnetworks.data.database.DataItem;
import net.snurkabill.neuralnetworks.data.database.Database;
import net.snurkabill.neuralnetworks.data.database.LabelledItem;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

import static org.junit.Assert.assertEquals;

public class DatabaseTest {

    public static final Logger LOGGER = LoggerFactory.getLogger("Database test");

    public static final int numberOfClasses = 3;
    public static final int[] sizeOfElementsTesting = {3, 7, 9};
    public static final int[] sizeOfElementsTraining = {8, 6, 3};
    public static final double tolerance = 0.0000001;

    private static Database database;

    private static int sumElements(int[] array) {
        int sum = 0;
        for (int i = 0; i < array.length; i++) {
            sum += array[i];
        }
        return sum;
    }

    public static Map<Integer, List<LabelledItem>> createTestingSet() {
        Map<Integer, List<LabelledItem>> map = new HashMap<>();
        for (int i = 0; i < numberOfClasses; i++) {
            List<LabelledItem> tmp = new ArrayList<>();
            for (int j = sizeOfElementsTesting[i] - 1; j >= 0; j--) {
                double[] vector = new double[1];
                vector[0] = i * 10 + j;
                tmp.add(new LabelledItem(new DataItem(vector), i));
            }
            map.put(i, tmp);
        }
        return map;
    }

    public static Map<Integer, List<LabelledItem>> createTrainingSet() {
        Map<Integer, List<LabelledItem>> map = new HashMap<>();
        for (int i = 0; i < numberOfClasses; i++) {
            List<LabelledItem> tmp = new ArrayList<>();
            for (int j = sizeOfElementsTraining[i] - 1; j >= 0; j--) {
                double[] vector = new double[1];
                vector[0] = i * 10 + j;
                tmp.add(new LabelledItem(new DataItem(vector), i));
            }
            map.put(i, tmp);
        }
        return map;
    }

    public static Map<Integer, List<LabelledItem>> createSimpleSet() {
        Map<Integer, List<LabelledItem>> trainingSet = new HashMap<>();
        List<LabelledItem> first = new ArrayList<>();
        first.add(new LabelledItem(new double[] {1, 1, 1}, 0));
        first.add(new LabelledItem(new double[] {2, 2, 2}, 0));
        first.add(new LabelledItem(new double[] {3, 3, 3}, 0));
        first.add(new LabelledItem(new double[] {100, 200, 300}, 0));
        List<LabelledItem> second = new ArrayList<>();
        second.add(new LabelledItem(new double[] {-1, -1, -1}, 1));
        second.add(new LabelledItem(new double[] {-2, -2, -2}, 1));
        second.add(new LabelledItem(new double[] {-3, -3, -3}, 1));
        second.add(new LabelledItem(new double[] {-100, -200, -300}, 1));
        trainingSet.put(0, first);
        trainingSet.put(1, second);
        return trainingSet;
    }

    @BeforeClass
    public static void beforeClass() {
        database = new Database(0, createTrainingSet(), createTestingSet(), "testingBase", true);
    }

    @Test
    public void testNumberOfElements() {
        assertEquals(numberOfClasses, database.getNumberOfClasses());
    }

    @Test
    public void testSizeOfSets() {
        for (int i = 0; i < database.getNumberOfClasses(); i++) {
            assertEquals(sizeOfElementsTesting[i], database.getSizeOfTestingDataset(i));
            assertEquals(sizeOfElementsTraining[i], database.getSizeOfTrainingDataset(i));
        }
        int sum = 0;
        for (int i = 0; i < sizeOfElementsTesting.length; i++) {
            sum += sizeOfElementsTesting[i];
        }
        assertEquals(sum, database.getTestSetSize());
        sum = 0;
        for (int i = 0; i < sizeOfElementsTraining.length; i++) {
            sum += sizeOfElementsTraining[i];
        }
        assertEquals(sum, database.getTrainingsetSize());
    }

    @Test
    public void testSizeOfVector() {
        assertEquals(1, database.getSizeOfVector());
    }

    @Test
    public void testPickedTestingIterator() {
        for (int i = 0; i < numberOfClasses; i++) {
            Iterator<DataItem> iterator = database.getTestingIteratorOverClass(i);
            for (int j = 0; iterator.hasNext(); j++) {
                double val = iterator.next().data[0];
                assertEquals(i * 10 + j, val, tolerance);
            }
        }
    }

    @Test
    public void testInfiniteTrainingIterator() {
        Iterator<DataItem> iterator = database.getInfiniteTrainingIterator();
        int lastFirst = 0;
        int lastSecond = 0;
        int lastThird = 0;
        for (int i = 0; i < sumElements(sizeOfElementsTraining) * 10; i++) {
            DataItem item = iterator.next();
            int num = 0;
            if (i % 3 == 0) {
                num = (i % 3) * 10 + lastFirst;
                lastFirst++;
                lastFirst = lastFirst % sizeOfElementsTraining[0];
            } else if (i % 3 == 1) {
                num = (i % 3) * 10 + lastSecond;
                lastSecond++;
                lastSecond = lastSecond % sizeOfElementsTraining[1];
            } else if (i % 3 == 2) {
                num = (i % 3) * 10 + lastThird;
                lastThird++;
                lastThird = lastThird % sizeOfElementsTraining[2];
            } else {
                throw new IllegalArgumentException("WTF");
            }
            assertEquals(item.data[0], num, tolerance);
        }
    }

    @Test
    public void unityBasedNormalization() {
        Database database = new Database(0, createSimpleSet(), createSimpleSet(), "database", false);
        Iterator<DataItem> iterator = database.getTestingIterator();

        double max1 = Double.NEGATIVE_INFINITY;
        double min1 = Double.POSITIVE_INFINITY;
        double max2 = Double.NEGATIVE_INFINITY;
        double min2 = Double.POSITIVE_INFINITY;
        double max3 = Double.NEGATIVE_INFINITY;
        double min3 = Double.POSITIVE_INFINITY;

        for (int i = 0; iterator.hasNext(); i++) {
            DataItem item = iterator.next();

            if(item.data[0] < min1) {
                min1 = item.data[0];
            }
            if(item.data[1] < min2) {
                min2 = item.data[1];
            }
            if(item.data[2] < min3) {
                min3 = item.data[2];
            }

            if(item.data[0] > max1) {
                max1 = item.data[0];
            }
            if(item.data[1] < max2) {
                max2 = item.data[1];
            }
            if(item.data[2] < max3) {
                max3 = item.data[2];
            }
        }
        assertEquals(10, min1, tolerance);
        assertEquals(10, min2, tolerance);
        assertEquals(10, min3, tolerance);
        assertEquals(20, max1, tolerance);
        assertEquals(20, max2, tolerance);
        assertEquals(20, max3, tolerance);
    }

    @Test
    @Ignore
    public void testPerformanceOfSingleAndMultipleThreadTestIterator() {
        Map<Integer, List<Integer>> testingSet = new HashMap<>();

        int numOfClasses = 100;
        int numOfElementsInclass = 100_000;

        for (int i = 0; i < numOfClasses; i++) {
            List<Integer> tmpList = new ArrayList<>();
            for (int j = 0; j < numOfElementsInclass; j++) {
                tmpList.add(j);
            }
            testingSet.put(i, tmpList);
        }
        Database tmpDatabase = new Database(0, null, testingSet, "testingPerformance", true);

        Iterator<DataItem> iterator = tmpDatabase.getTestingIterator();

        long startTime = System.currentTimeMillis();

        for (; iterator.hasNext(); ) {
            DataItem a = iterator.next();
            if (a.data[0] == -1) {
                LOGGER.info("asdf");
            }
        }
        long endTime = System.currentTimeMillis();
        LOGGER.info("Time spent: {}", endTime - startTime);
    }
}

