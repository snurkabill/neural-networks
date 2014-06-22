package net.snurkabill.neuralnetworks.energybasednetwork;

public abstract class EnergyBased {
	
	protected double energy;

	public abstract double calcEnergy();
	
	public final double getEnergy() {
		return energy;
	}
}
