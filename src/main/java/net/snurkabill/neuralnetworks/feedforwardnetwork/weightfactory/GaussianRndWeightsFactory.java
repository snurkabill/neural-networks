package net.snurkabill.neuralnetworks.feedforwardnetwork.weightfactory;

import java.util.Random;

public class GaussianRndWeightsFactory implements WeightsFactory {

	private final double weightScale;
	private final Random random = new Random();
	
	public GaussianRndWeightsFactory(double weightScale, long seed) {
		this.weightScale = weightScale;
		this.random.setSeed(seed);
	}
	
	@Override
	public void setWeights(double[][][] weights) {
		for (double[][] weight : weights) {
			setWeights(weight);
		}
	}
	
	@Override
	public void setWeights(double[][] weights) {
		for (double[] weight : weights) {
			setWeights(weight);
		}
	}
	
	@Override
	public void setWeights(double[] weights) {
		for (int k = 0; k < weights.length; k++) {
			weights[k] = random.nextGaussian() * weightScale;
		}
	}
}
