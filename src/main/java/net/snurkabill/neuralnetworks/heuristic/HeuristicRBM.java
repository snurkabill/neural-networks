package net.snurkabill.neuralnetworks.heuristic;

public class HeuristicRBM extends Heuristic {

    public double temperature;
    public int numOfTrainingIterations;
    public int constructiveDivergenceIndex;

    @Deprecated
    public static HeuristicRBM createBasicHeuristicParams() {
        HeuristicRBM rbm = new HeuristicRBM();
        rbm.temperature = 1;
        rbm.constructiveDivergenceIndex = 3;
        rbm.numOfTrainingIterations = 30;
        rbm.momentum = 0.3;
        rbm.learningRate = 0.1;
        return rbm;
    }

    public static HeuristicRBM createStartingHeuristicParams() {
        HeuristicRBM heuristic = new HeuristicRBM();
        heuristic.constructiveDivergenceIndex = 1;
        heuristic.numOfTrainingIterations = 5;
        heuristic.temperature = 1;
        heuristic.learningRate = 0.2;
        heuristic.momentum = 0.5;
        return heuristic;
    }

    @Override
    public String toString() {
        return super.toString() + " temperature: " + String.valueOf(temperature) +
                " sizeOfVectorBatch: " + String.valueOf(numOfTrainingIterations) +
                " constructiveDivergence: " + String.valueOf(constructiveDivergenceIndex);
    }
}
