package net.snurkabill.neuralnetworks.weights.weightfactory;

public interface WeightsFactory {

    public void setWeights(double[][][] weights);

    public void setWeights(double[][] weights);

    public void setWeights(double[] weights);
}
