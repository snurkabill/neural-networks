package net.snurkabill.neuralnetworks.data;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Database<T extends DataItem> {
	
	private static final Logger LOGGER = LoggerFactory.getLogger("Database");
	
	private final Random random;
	
	private final Map<Integer, List<T>> trainingSet;
	private final Map<Integer, List<T>> testingSet;
	private final Map<Integer, Iterator<T>> trainingSetIterator = new HashMap<>();
	private final Map<Integer, Iterator<T>> testingSetIterator = new HashMap<>();
	
	private int sizeOfTrainingSet;
	private int sizeOfTestingSet;
	private int numberOfClasses;
	private final String name;
	private final int sizeOfVector;
	
	public Database(long seed, Map<Integer, List<T>> trainingSet, Map<Integer, List<T>> testingSet, String name) {
		
		if(trainingSet.size() != testingSet.size()) {
			throw new IllegalArgumentException("TrainingSet has different size than testingSet");
		}
		this.name = name;
		this.random = new SecureRandom();
		this.random.setSeed(seed);
		this.trainingSet = trainingSet;
		this.testingSet = testingSet;
		this.numberOfClasses = trainingSet.size();
		for (int i = 0; i < trainingSet.size(); i++) {
			sizeOfTrainingSet += trainingSet.get(i).size();
			sizeOfTestingSet += testingSet.get(i).size();
			testingSetIterator.put(i, testingSet.get(i).iterator());
			trainingSetIterator.put(i, trainingSet.get(i).iterator());
		}
		sizeOfVector = trainingSet.get(0).get(0).data.length;
	}
	
	public T getTrainingIteratedData(int i) {
		if(!trainingSetIterator.get(i).hasNext()) {
			LOGGER.debug("Setting new iterator of trainingSet for {}th list", i);
			trainingSetIterator.put(i, trainingSet.get(i).iterator());
		}
		return trainingSetIterator.get(i).next();
	}
	
	public T getTestingIteratedData(int i) {
		if(!testingSetIterator.get(i).hasNext()) {
			LOGGER.debug("Setting new iterator of testingSet for {}th list", i);
			testingSetIterator.put(i, testingSet.get(i).iterator());
		}
		return testingSetIterator.get(i).next();
	}
	
	public T getTrainingData(int i) {
		List<T> list = trainingSet.get(i);
		return list.get(random.nextInt(list.size()));
	}
	
	public T getTestingData(int i) {
		List<T> list = testingSet.get(i);
		return list.get(random.nextInt(list.size()));
	}
	
	// TODO : randomized data with labels!
	public T getRandomizedTestingData() {
		return getTestingData(random.nextInt(testingSet.size()));
	}
	
	public T getRandomizedTrainingData() {
		return getTrainingData(random.nextInt(trainingSet.size()));
	}
	
	public int getTestsetSize() {
		return sizeOfTestingSet;
	}
	
	public int getTrainingsetSize() {
		return sizeOfTrainingSet;
	}

	public int getNumberOfClasses() {
		return numberOfClasses;
	}
	
	public int getSizeOfTestingDataset(int index) {
		return testingSet.get(index).size();
	}
	
	public int getSizeOfTrainingDataset(int index) {
		return trainingSet.get(index).size();
	}

	public int getSizeOfVector() {
		return sizeOfVector;
	}
	
	public void storeDatabase() throws FileNotFoundException, IOException {
		try (DataOutputStream os = new DataOutputStream(new FileOutputStream(new File(name)))) {
			os.writeInt(numberOfClasses);
			os.writeInt(trainingSet.get(0).get(0).data.length);
			for (int i = 0; i < numberOfClasses; i++) {
				os.writeInt(trainingSet.get(i).size());
			}
			for (int i = 0; i < numberOfClasses; i++) {
				for (int j = 0; j < trainingSet.get(i).size(); j++) {
					for (int k = 0; k < trainingSet.get(0).get(0).data.length; k++) {
						os.writeDouble(trainingSet.get(i).get(j).data[k]);
					}
				}
			}
			for (int i = 0; i < numberOfClasses; i++) {
				os.writeInt(testingSet.get(i).size());
			}
			for (int i = 0; i < numberOfClasses; i++) {
				for (int j = 0; j < testingSet.get(i).size(); j++) {
					for (int k = 0; k < testingSet.get(0).get(0).data.length; k++) {
						os.writeDouble(testingSet.get(i).get(j).data[k]);
					}
				}
			}
			os.flush();
		}
	}
	
	public static Database loadDatabase(File baseFile) throws FileNotFoundException, IOException {
		return loadDatabase(baseFile, 0);
	}
	
	public static Database loadDatabase(File baseFile, long seed) throws FileNotFoundException, IOException {
		Map<Integer, List<DataItem>> trainingSet;
		Map<Integer, List<DataItem>> testingSet;
		try (DataInputStream is = new DataInputStream(new FileInputStream(baseFile))) {
			trainingSet = new HashMap<>();
			testingSet = new HashMap<>();
			int numberOfClasses = is.readInt();
			for (int i = 0; i < numberOfClasses; i++) {
				trainingSet.put(i, new ArrayList<DataItem>());
				testingSet.put(i, new ArrayList<DataItem>());
			}	int sizeOfVector = is.readInt();
			List<Integer> sizesOfTrainingLists = new ArrayList<>();
			for (int i = 0; i < numberOfClasses; i++) {
				sizesOfTrainingLists.add(is.readInt());
			}	double[] buffer = new double[sizeOfVector];
			for (int i = 0; i < numberOfClasses; i++) {
				for (int j = 0; j < sizesOfTrainingLists.get(i); j++) {
					for (int k = 0; k < sizeOfVector; k++) {
						buffer[k] = is.readDouble();
					}
					trainingSet.get(i).add(new DataItem(buffer));
				}
			}
			List<Integer> sizesOfTestingLists = new ArrayList<>();
			for (int i = 0; i < numberOfClasses; i++) {
				sizesOfTestingLists.add(is.readInt());
			}
			for (int i = 0; i < numberOfClasses; i++) {
				for (int j = 0; j < sizesOfTestingLists.get(i); j++) {
					for (int k = 0; k < sizeOfVector; k++) {
						buffer[k] = is.readDouble();
					}
					testingSet.get(i).add(new DataItem(buffer));
				}
			}
		}
		return new Database(seed, trainingSet, testingSet, baseFile.getName());
	}
}
