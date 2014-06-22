package net.snurkabill.neuralnetworks.data;

import java.security.SecureRandom;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Database<T extends DataItem> {
	
	private static final Logger LOGGER = LoggerFactory.getLogger("Database");
	
	private final SecureRandom random;
	
	private final Map<Integer, List<T>> trainingSet;
	private final Map<Integer, List<T>> testingSet;
	
	private final Map<Integer, Iterator<T>> trainingSetIterator = new HashMap<>();
	private final Map<Integer, Iterator<T>> testingSetIterator = new HashMap<>();
	
	private int sizeOfTrainingSet;
	private int sizeOfTestingSet;

	private int numberOfClasses;
	
	public Database(long seed, Map<Integer, List<T>> trainingSet, Map<Integer, List<T>> testingSet) {
		
		if(trainingSet.size() != testingSet.size()) {
			throw new IllegalArgumentException("TrainingSet has different size than testingSet");
		}
		
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
	
	public T randomizedGetTestingData() {
		return getTestingData(random.nextInt(testingSet.size()));
	}
	
	public T randomizedGetTrainingData() {
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
}
