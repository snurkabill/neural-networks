package net.snurkabill.neuralnetworks.neuralnetwork;

public interface FeedForwardable {

    public void feedForward(double[] inputVector);

    public double[] getForwardedValues();
}
