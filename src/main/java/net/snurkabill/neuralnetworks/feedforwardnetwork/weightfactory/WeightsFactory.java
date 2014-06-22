package net.snurkabill.neuralnetworks.feedforwardnetwork.weightfactory;

public interface WeightsFactory {
	
	public void setWeights(double[][][] weights);
	public void setWeights(double[][] weights);
	public void setWeights(double[] weights);
}
