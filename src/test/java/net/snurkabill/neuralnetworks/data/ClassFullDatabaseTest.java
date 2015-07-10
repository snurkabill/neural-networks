package net.snurkabill.neuralnetworks.data;

import net.snurkabill.neuralnetworks.data.database.ClassFullDatabase;
import net.snurkabill.neuralnetworks.data.database.DataItem;
import net.snurkabill.neuralnetworks.data.database.LabelledItem;
import org.junit.BeforeClass;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;

public class ClassFullDatabaseTest {

    public static final Logger LOGGER = LoggerFactory.getLogger("ClassFullDatabase test");

    public static final int numberOfClasses = 3;
    public static final int[] sizeOfElementsTesting = {3, 7, 9};
    public static final int[] sizeOfElementsTraining = {8, 6, 3};
    public static final double tolerance = 0.0000001;

    private static ClassFullDatabase classFullDatabase;

    private static int sumElements(int[] array) {
        int sum = 0;
        for (int anArray : array) {
            sum += anArray;
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

    @BeforeClass
    public static void beforeClass() {
        classFullDatabase = new ClassFullDatabase(0, createTrainingSet(), createTestingSet(), "testingBase", true);
    }

    @Test
    public void testNumberOfElements() {
        assertEquals(numberOfClasses, classFullDatabase.getNumberOfClasses());
    }

    @Test
    public void testSizeOfSets() {
        for (int i = 0; i < classFullDatabase.getNumberOfClasses(); i++) {
            assertEquals(sizeOfElementsTesting[i], classFullDatabase.getSizeOfTestingDataset(i));
            assertEquals(sizeOfElementsTraining[i], classFullDatabase.getSizeOfTrainingDataset(i));
        }
        int sum = 0;
        for (int aSizeOfElementsTesting : sizeOfElementsTesting) {
            sum += aSizeOfElementsTesting;
        }
        assertEquals(sum, classFullDatabase.getTestSetSize());
        sum = 0;
        for (int aSizeOfElementsTraining : sizeOfElementsTraining) {
            sum += aSizeOfElementsTraining;
        }
        assertEquals(sum, classFullDatabase.getTrainingSetSize());
    }

    @Test
    public void testSizeOfVector() {
        assertEquals(1, classFullDatabase.getSizeOfVector());
    }

    @Test
    public void testPickedTestingIterator() {
        for (int i = 0; i < numberOfClasses; i++) {
            ClassFullDatabase.TestClassIterator iterator = classFullDatabase.getTestingIteratorOverClass(i);
            for (int j = 0; iterator.hasNext(); j++) {
                double val = iterator.next().data[0];
                assertEquals(i * 10 + j, val, tolerance);
            }
        }
    }

    @Test
    public void testInfiniteTrainingIterator() {
        ClassFullDatabase.InfiniteSimpleTrainingIterator iterator = classFullDatabase.getInfiniteTrainingIterator();
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
    public void storeLoadTest() throws IOException {
        classFullDatabase.storeDatabase();
        ClassFullDatabase classFullDatabase2 = new ClassFullDatabase(0, new File("testingBase"));

        assertEquals(classFullDatabase.getNumberOfClasses(), classFullDatabase2.getNumberOfClasses());
        assertEquals(classFullDatabase.getSizeOfVector(), classFullDatabase2.getSizeOfVector());
        assertEquals(classFullDatabase.getTestSetSize(), classFullDatabase2.getTestSetSize());
        assertEquals(classFullDatabase.getTrainingSetSize(), classFullDatabase2.getTrainingSetSize());
        assertEquals(classFullDatabase.getName(), classFullDatabase2.getName());

        for (int i = 0; i < classFullDatabase.getNumberOfClasses(); i++) {
            assertEquals(classFullDatabase.getSizeOfTestingDataset(i), classFullDatabase2.getSizeOfTestingDataset(i));
            assertEquals(classFullDatabase.getSizeOfTrainingDataset(i), classFullDatabase2.getSizeOfTrainingDataset(i));
        }

        ClassFullDatabase.InfiniteSimpleTrainingIterator originalIterator = classFullDatabase.getInfiniteTrainingIterator();
        ClassFullDatabase.InfiniteSimpleTrainingIterator copyIterator = classFullDatabase2.getInfiniteTrainingIterator();

        for (int i = 0; i < classFullDatabase.getTrainingSetSize() * 2; i++) {
            LabelledItem item1 = originalIterator.next();
            LabelledItem item2 = copyIterator.next();
            assertEquals(item1._class, item2._class);
            assertEquals(item1.data[0], item2.data[0], tolerance);
        }

        for (int i = 0; i < classFullDatabase.getNumberOfClasses(); i++) {
            ClassFullDatabase.TestClassIterator originalTestIterator = classFullDatabase.getTestingIteratorOverClass(i);
            ClassFullDatabase.TestClassIterator copyTestIterator = classFullDatabase2.getTestingIteratorOverClass(i);
            for (int j = 0; j < classFullDatabase.getSizeOfTestingDataset(i); j++) {
                assertEquals(originalTestIterator.next().data[0], copyTestIterator.next().data[0], tolerance);
            }
        }
        Files.delete(new File("testingBase").toPath());
    }

}

