package net.snurkabill.neuralnetworks.data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

public class DatabaseTest {
	
	public static final int numberOfClasses = 3;
	public static final int[] sizeOfElementsTesting = {3, 7, 10};
	public static final int[] sizeOfElementsTraining = {8, 6, 3};
	
	private static Database database;
	
	public static Map<Integer, List<LabelledItem>> createTestingSet() {
		Map<Integer, List<LabelledItem>> map = new HashMap<>();
		for (int i = 0; i < numberOfClasses; i++) {
			List<LabelledItem> tmp = new ArrayList<>();
			for (int j = 0; j < sizeOfElementsTesting[i]; j++) {
				double[] vector = new double[1];
				vector[0] = i*10 + j;
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
			for (int j = 0; j < sizeOfElementsTraining[i]; j++) {
				double[] vector = new double[1];
				vector[0] = i*10 + j;
				tmp.add(new LabelledItem(new DataItem(vector), i));
			}
			map.put(i, tmp);
		}
		return map;
	}
	
	@BeforeClass
	public static void beforeClass() {
		database = new Database(0, createTrainingSet(), createTestingSet(), "testingBase");
	}
	
	@Test
	public void testNumberOfElements() {
		Assert.assertEquals(numberOfClasses, database.getNumberOfClasses());
	}
	
	@Test
	public void testSizeOfSets() {	
		for (int i = 0; i < database.getNumberOfClasses(); i++) {
			Assert.assertEquals(sizeOfElementsTesting[i], database.getSizeOfTestingDataset(i));
			Assert.assertEquals(sizeOfElementsTraining[i], database.getSizeOfTrainingDataset(i));
		}
		int sum = 0;
		for (int i = 0; i < sizeOfElementsTesting.length; i++) {
			sum += sizeOfElementsTesting[i];
		}
		Assert.assertEquals(sum, database.getTestsetSize());
		sum = 0;
		for (int i = 0; i < sizeOfElementsTraining.length; i++) {
			sum += sizeOfElementsTraining[i];
		}
		Assert.assertEquals(sum, database.getTrainingsetSize());
	}
	
	@Test
	public void testSizeOfVector() {
		Assert.assertEquals(1, database.getSizeOfVector());
	}
}

