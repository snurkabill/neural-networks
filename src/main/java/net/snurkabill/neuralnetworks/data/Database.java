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
	
	public LabelledItem getRandomizedLabelTestingData() {
		int _class = random.nextInt(testingSet.size());
		return new LabelledItem(getTestingData(_class), _class);
	}
	
	public LabelledItem getRandomizedLabelTrainingData() {
		int _class = random.nextInt(trainingSet.size());
		return new LabelledItem(getTrainingData(_class), _class);
	}
	
	public T getRandomizedTestingData() {
		return getTestingData(random.nextInt(testingSet.size()));
	}
	
	public T getRandomizedTrainingData() {
		return getTrainingData(random.nextInt(trainingSet.size()));
	}
	
	public int getTestSetSize() {
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
		LOGGER.info("Storing database: {}", name);
		long time1 = System.currentTimeMillis();
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
		long time2 = System.currentTimeMillis();
		LOGGER.info("Storing database done, took: {}", time2 - time1);
	}
	
	public static Database loadDatabase(File baseFile) throws FileNotFoundException, IOException {
		return loadDatabase(baseFile, 0);
	}
	
	public static Database loadDatabase(File baseFile, long seed) throws FileNotFoundException, IOException {
		LOGGER.info("Loading database {}", baseFile.getName());
		long time1 = System.currentTimeMillis();
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
		long time2 = System.currentTimeMillis();
		LOGGER.info("Loading database done, took: {}", time2 - time1);
		return new Database(seed, trainingSet, testingSet, baseFile.getName());
	}
	
	public synchronized Iterator<T> getTestingIteratorOverClass(int i) {
		return testingSet.get(i).iterator();
	}
	
	public synchronized Iterator<T> getTrainingIteratorOverClass(int i) {
		return trainingSet.get(i).iterator();
	}
	
	public synchronized Iterator<T> getTestingIterator() {
		return new TestSetIterator(testingSet);
	}
	
	public synchronized Iterator<T> getInfiniteTrainingIterator() {
		return new InfiniteSimpleTrainSetIterator(trainingSet);
	}
	
	public class InfiniteSimpleTrainSetIterator implements Iterator {

		private Map<Integer, List<T>> map;
		private List<Iterator<T>> iterators = new ArrayList<>();
		private int lastUnusedClass;
		
		public InfiniteSimpleTrainSetIterator(Map<Integer, List<T>> map) {
			this.map = map;
			for (int i = 0; i < map.size(); i++) {
				iterators.add(map.get(i).iterator());
			}

			lastUnusedClass = 0;
			// FINISH ME
		}

		@Override
		public boolean hasNext() {
			return true;
		}

		@Override
		public LabelledItem next() {
			if(lastUnusedClass == map.size()) {
				lastUnusedClass = 0;
			}
			if(!iterators.get(lastUnusedClass).hasNext()) {
				iterators.set(lastUnusedClass, map.get(lastUnusedClass).iterator());
			}
			lastUnusedClass++;
			return new LabelledItem(iterators.get(lastUnusedClass - 1).next(), lastUnusedClass - 1);
		}	


		@Override
		public void remove() {
			throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
		}	
	}
	
	public class TestSetIterator implements Iterator {
		private List<Iterator<T>> iterators = new ArrayList<>();
		private Iterator<Iterator<T>> master;
		private Iterator<T> actual;
		private int actualClass;
		
		public TestSetIterator(Map<Integer, List<T>> map) {
			for (int i = 0; i < map.size(); i++) {
				iterators.add(map.get(i).iterator());
			}
			master = iterators.iterator();
			actual = master.next();
			actualClass = 0;
		}
		
		@Override
		public boolean hasNext() {
			if(master.hasNext()) {
				return true;
			} else return actual.hasNext() == true;
		}

		@Override
		public LabelledItem next() {
			if(actual.hasNext()) {
				return new LabelledItem(actual.next(), actualClass);
			} else {
				actual = master.next();
				actualClass++;
				return new LabelledItem(actual.next(), actualClass);
			}
		}

		@Override
		public void remove() {
			throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
		}
	}
}
