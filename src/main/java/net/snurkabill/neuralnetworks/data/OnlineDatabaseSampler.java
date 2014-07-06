package net.snurkabill.neuralnetworks.data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class OnlineDatabaseSampler {
	
	private static final Logger LOGGER = LoggerFactory.getLogger("Online Database Sampler");
	
	private final Map<Integer, List<DataItem>> trainingSet;
	private final Map<Integer, List<DataItem>> testingSet;
	private final List<Integer> sumsOfVectors = new ArrayList<>();
	
	/*private final Map<Integer, List<DataItem>> vectorsBuffer = new HashMap<>();
	private final int maxSizeOfBuffer;
	*/
	private final int classesOfDivision;
	private final int trainTestRatio;
	private int allVectors = 0;
	private int classWithLeastVectors;
	private int classRatio;
	private final int sizeOfVector;
	
	private final String databaseName;

	public OnlineDatabaseSampler(String databaseName, int sizeOfVector, int classesOfDivision, 
			int ratioOfMaxDiffBetweenClasses, int trainTestRatio) {
		if(ratioOfMaxDiffBetweenClasses <= 0) {
			throw new IllegalArgumentException("Invalid ratio");
		}
		if(trainTestRatio <= 1) {
			throw new IllegalArgumentException("Invalid train test ratio");
		}
		this.sizeOfVector = sizeOfVector;
		this.classesOfDivision = classesOfDivision;
		this.classRatio = ratioOfMaxDiffBetweenClasses;
		this.trainTestRatio = trainTestRatio;
		this.databaseName = databaseName;
		trainingSet = new HashMap<>(classesOfDivision);
		testingSet = new HashMap<>(classesOfDivision);
		/*for (int i = 0; i < classesOfDivision; i++) {
			vectorsBuffer.put(i, new LinkedList<DataItem>());
		}*/
		for (int i = 0; i < classesOfDivision; i++) {
			trainingSet.put(i, new ArrayList<DataItem>());
			testingSet.put(i, new ArrayList<DataItem>());
		}
		for (int i = 0; i < classesOfDivision; i++) {
			sumsOfVectors.add(1);
		}
	}

	public String getName() {
		return databaseName;
	}

	public Map<Integer, List<DataItem>> getTrainingSet() {
		return trainingSet;
	}

	public Map<Integer, List<DataItem>> getTestingSet() {
		return testingSet;
	}

	public int getRatio() {
		return trainTestRatio;
	}

	public int getAllVectors() {
		return allVectors;
	}
	
	public void sampleVector(int classID, double[] inputVector) {
		if(classID == classWithLeastVectors) {
			addVector(classID, inputVector);
			calcLeastVectorIndex();
		} else if(sumsOfVectors.get(classID) / sumsOfVectors.get(classWithLeastVectors) <= classRatio) {
			addVector(classID, inputVector);
		} else {
			LOGGER.debug("Skiping vector of ID: {}", classID);
			LOGGER.debug("Class with least vectors {}", classWithLeastVectors);
		}
	}
	
	private void calcLeastVectorIndex() {
		int leastVectorsIndex = 0;
		int min = sumsOfVectors.get(0);
		for (int i = 0; i < classesOfDivision; i++) {
			if(sumsOfVectors.get(i) < min) {
				min = sumsOfVectors.get(i);
				leastVectorsIndex = i;
			}
		}
		classWithLeastVectors = leastVectorsIndex;
	}
	
	private void addVector(int classID, double[] inputVector) {
		allVectors++;
		sumsOfVectors.set(classID, sumsOfVectors.get(classID) + 1);
		if(sumsOfVectors.get(classID) % trainTestRatio == 0) {
			testingSet.get(classID).add(new DataItem(inputVector));
		} else {
			trainingSet.get(classID).add(new DataItem(inputVector));
		}
	}
	
	public int sizeOfDatabaseInMegabytes() {
		return (int)((sizeOfVector * allVectors * 8) / 1_000_000);
	}
	
	public Database getDatabase(String name, long seed) {
		return new Database(seed, trainingSet, testingSet, name);
	}
	
	public Database getDatabase(String name) {
		return getDatabase(name, 0);
	}
}
