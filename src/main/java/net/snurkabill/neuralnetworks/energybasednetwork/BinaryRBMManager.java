package net.snurkabill.neuralnetworks.energybasednetwork;

import net.snurkabill.neuralnetworks.benchmark.RBM.Vector.Tuple;
import net.snurkabill.neuralnetworks.benchmark.RBM.VectorAssociationMeter;
import net.snurkabill.neuralnetworks.data.DataItem;
import net.snurkabill.neuralnetworks.data.Database;
import net.snurkabill.neuralnetworks.energybasednetwork.randomize.randomgenerator.RandomBinaryDimensionsGenerator;
import net.snurkabill.neuralnetworks.energybasednetwork.randomize.randomindexer.LinearRandomIndexer;
import net.snurkabill.neuralnetworks.energybasednetwork.randomize.randomindexer.RandomDimensionsIndexer;
import net.snurkabill.neuralnetworks.results.UnsupervisedTestResults;
import net.snurkabill.neuralnetworks.utilities.Utilities;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class BinaryRBMManager {
	
	private static final Logger LOGGER = LoggerFactory.getLogger("RBM binary manager");
	
	private final List<BinaryRestrictedBoltzmannMachine> rbms;
	private final Database database;
    private final int sizeOfInputVector;
    private final long seed;
	
	public BinaryRBMManager(List<BinaryRestrictedBoltzmannMachine> networks, Database database, long seed) {
		this.rbms = networks;
        this.sizeOfInputVector = networks.get(0).getSizeOfVisibleVector();
        for (BinaryRestrictedBoltzmannMachine rbm : networks) {
            if(sizeOfInputVector != rbm.getSizeOfVisibleVector()) {
                throw new IllegalArgumentException("Input Vector Sizes are different!");
            }
        }
		this.database = database;
        this.seed = seed;
		LOGGER.info("Created Binary RBM manager");
	}

	public void trainNetworks(int numOfIterations) {
		if(numOfIterations <= 0) {
			throw new IllegalArgumentException("Wrong NumOfIterations");
		}
		for (int network = 0; network < rbms.size(); network++) {
			long trainingStarted = System.currentTimeMillis();
			for (int i = 0; i < numOfIterations; i++) {
				rbms.get(network).trainMachine(database.getRandomizedTrainingData().data);
			}
			long trainingEnded = System.currentTimeMillis();
			double sec = ((trainingEnded - trainingStarted) / 1000.0);
			LOGGER.info("Training: {} samples took {} seconds, {} samples/sec", numOfIterations, sec, numOfIterations/sec);	
		}
	}

	public List<UnsupervisedTestResults> testNetworks(RandomDimensionsIndexer randomIndexer) {
        int numOfUnknownDimensionsOfVector = randomIndexer.getNumberOfUnknown();
        if(numOfUnknownDimensionsOfVector >= this.sizeOfInputVector) {
            throw new IllegalArgumentException("Vector must have some known dimensions for association");
        }
        if(numOfUnknownDimensionsOfVector < 0) {
            throw new IllegalArgumentException("There can't be negative num of unknown dimensions of vector");
        }
		List<UnsupervisedTestResults> results = new ArrayList<>();
        RandomBinaryDimensionsGenerator generator = new RandomBinaryDimensionsGenerator(sizeOfInputVector, seed);
        for (BinaryRestrictedBoltzmannMachine machine : rbms) {
            double distributedSum = 0.0;
            long testingStarted = System.currentTimeMillis();
            Iterator<DataItem> iterator = database.getTestingIterator();
            int numOfVectors = database.getTestSetSize();
            for (; iterator.hasNext();) {
                Tuple item = new Tuple(iterator.next().data);
                boolean[] knownIndexes = randomIndexer.generateIndexes();
                machine.setVisibleNeurons(generator.generateVector(item.getDoubleArray(), knownIndexes));
                machine.activateHiddenNeurons();
                Tuple resultTuple = new Tuple(machine.activateVisibleNeurons());
                distributedSum += VectorAssociationMeter.check(item, resultTuple, knownIndexes,
                        numOfUnknownDimensionsOfVector);
            }
            distributedSum /= numOfVectors;
            long testingEnded = System.currentTimeMillis();
            results.add(new UnsupervisedTestResults(numOfVectors, machine.calcEnergy(),
                    testingEnded - testingStarted, distributedSum,
                    (numOfUnknownDimensionsOfVector/(double)this.sizeOfInputVector) * 100));
            double sec = Utilities.secondsSpent(testingStarted, testingEnded);
            LOGGER.info("Testing: {} samples took {} seconds, {} samples/sec", numOfVectors, sec, numOfVectors/sec);
        }
        return results;
	}
}
