package net.snurkabill.neuralnetworks.heuristic;

public class HeuristicRBM extends Heuristic {

    public double temperature;
    public int numOfTrainingIterations;
    public int constructiveDivergenceIndex;

    public static HeuristicRBM createBasicHeuristicParams() {
        HeuristicRBM rbm = new HeuristicRBM();
        rbm.temperature = 1;
        rbm.constructiveDivergenceIndex = 3;
        rbm.numOfTrainingIterations = 30;
        rbm.momentum = 0.3;
        rbm.learningRate = 0.1;
        return rbm;
    }

    @Override
    public String toString() {
        return super.toString() + "temperature: " + String.valueOf(temperature) +
                " oneVectorTrainingIterations: " + String.valueOf(numOfTrainingIterations) +
                " constructiveDivergenceIndex: " + String.valueOf(constructiveDivergenceIndex);
    }
}
