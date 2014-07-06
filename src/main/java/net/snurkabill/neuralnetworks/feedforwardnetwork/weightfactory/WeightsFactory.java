package net.snurkabill.neuralnetworks.feedforwardnetwork.weightfactory;

public interface WeightsFactory {
	
	public void setWeights(double[][][] weights);
	public void setWeights(double[][] weights);
	public void setWeights(double[] weights);
	public void setWeights(int layerIndex, double[][] weights);
	public void setWeights(int layerIndex, int neuronIndex, double[] weights);
}
