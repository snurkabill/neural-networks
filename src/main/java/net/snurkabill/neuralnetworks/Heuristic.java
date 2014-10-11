package net.snurkabill.neuralnetworks;

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
}
