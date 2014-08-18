package net.snurkabill.neuralnetworks.energybasednetwork;

import java.util.List;
import net.snurkabill.neuralnetworks.data.Database;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class BinaryRBMManager {
	
	private static final Logger LOGGER = LoggerFactory.getLogger("RBM binary manager");
	
	private final List<BinaryRestrictedBoltzmannMachine> rbms;
	private final Database database;
	
	public BinaryRBMManager(List<BinaryRestrictedBoltzmannMachine> networks, Database database) {
		this.rbms = networks;
		this.database = database;
		LOGGER.info("Created Binary RBM manager");
	}

	public void trainNetwork(int numOfIterations) {
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
	
	public void testNetwork() {
		throw new UnsupportedOperationException("Not implemented yet");
	}
}
