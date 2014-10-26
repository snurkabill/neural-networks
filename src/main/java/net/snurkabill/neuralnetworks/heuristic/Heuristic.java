package net.snurkabill.neuralnetworks.heuristic;

public abstract class Heuristic {

    public double learningRate;
    public double momentum;
    private String name;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    @Override
    public String toString() {
        return "Heuristic: " + name + " learningRate: " + String.valueOf(learningRate) + "momentum" +
                String.valueOf(momentum);
    }
}
