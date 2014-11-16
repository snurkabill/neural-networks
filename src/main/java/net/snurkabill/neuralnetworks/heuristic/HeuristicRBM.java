package net.snurkabill.neuralnetworks.heuristic;

public class HeuristicRBM extends Heuristic {

    public double temperature;
    public int numOfTrainingIterations;
    public int constructiveDivergenceIndex;

    public HeuristicRBM() {
        this.temperature = 1;
        this.numOfTrainingIterations = 1;
        this.constructiveDivergenceIndex = 1;
    }

    @Deprecated
    public static HeuristicRBM createBasicHeuristicParams() {
        HeuristicRBM rbm = new HeuristicRBM();
        rbm.temperature = 1;
        rbm.constructiveDivergenceIndex = 3;
        rbm.numOfTrainingIterations = 10;
        rbm.momentum = 0.3;
        rbm.learningRate = 0.01;
        return rbm;
    }

    public static HeuristicRBM createStartingHeuristicParams() {
        HeuristicRBM heuristic = new HeuristicRBM();
        heuristic.constructiveDivergenceIndex = 1;
        heuristic.numOfTrainingIterations = 3;
        heuristic.temperature = 1;
        heuristic.learningRate = 0.1;
        heuristic.momentum = 0.3;
        return heuristic;
    }

    @Override
    public String toString() {
        return super.toString() + " temperature: " + String.valueOf(temperature) +
                " sizeOfVectorBatch: " + String.valueOf(numOfTrainingIterations) +
                " constructiveDivergence: " + String.valueOf(constructiveDivergenceIndex);
    }
}
