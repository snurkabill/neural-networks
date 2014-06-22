package net.snurkabill.neuralnetworks.energybasednetwork;

import net.snurkabill.neuralnetworks.utilities.Utilities;
import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class BoltzmannMachine extends EnergyBased {
	
	private static final Logger LOGGER = LoggerFactory.getLogger("Boltzmann Machine");
	
	protected double temperature = 1;
	
	protected final Random random;

	public BoltzmannMachine(long seed) {
		this.random = new Random(seed);
	}
	
	protected double calcProbabilityOfPositiveOutput(double potential) {
		return Utilities.sigmoid(potential / temperature);
	}
	
	public double getTemperature() {
		return temperature;
	}

	public void setTemperature(double temperature) {
		LOGGER.debug("Setting temperature to: {}", temperature);
		this.temperature = temperature;
	}
}
