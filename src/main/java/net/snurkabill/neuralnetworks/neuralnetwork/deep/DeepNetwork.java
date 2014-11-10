package net.snurkabill.neuralnetworks.neuralnetwork.deep;


import net.snurkabill.neuralnetworks.heuristic.Heuristic;
import net.snurkabill.neuralnetworks.neuralnetwork.NeuralNetwork;

public class DeepNetwork extends NeuralNetwork {


    public DeepNetwork(String name) {
        super(name);
    }

    @Override
    public void calculateNetwork(double[] inputVector) {
        
    }

    @Override
    protected double calcOutputVectorError(double[] targetValues) {
        return 0;
    }

    @Override
    public int getSizeOfOutputVector() {
        return 0;
    }

    @Override
    public int getSizeOfInputVector() {
        return 0;
    }

    @Override
    public double[] getOutputValues() {
        return new double[0];
    }

    @Override
    public void getOutputValues(double[] outputValues) {

    }

    @Override
    public void trainNetwork(double[] targetValues) {

    }

    @Override
    public Heuristic getHeuristic() {
        return null;
    }

    @Override
    public void setHeuristic(Heuristic heuristic) {

    }


}
