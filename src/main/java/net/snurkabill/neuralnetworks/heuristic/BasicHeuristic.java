package net.snurkabill.neuralnetworks.heuristic;


import com.thoughtworks.xstream.annotations.XStreamAlias;

@XStreamAlias("basicHeuristic")
public abstract class BasicHeuristic {

    private String name;

    // public for faster access
    public double learningRate;
    public double momentum;
    public double l2RegularizationConstant;
    public boolean dynamicKillingWeights;
    public int batchSize;

    public BasicHeuristic() {
        this.learningRate = 0.1;
        this.momentum = 0;
        this.dynamicKillingWeights = false;
        this.l2RegularizationConstant = 1;
        this.batchSize = 1;
    }

    public String getName() {
        return name;
    }

    public BasicHeuristic setName(String name) {
        this.name = name;
        return this;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public BasicHeuristic setLearningRate(double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    public double getMomentum() {
        return momentum;
    }

    public BasicHeuristic setMomentum(double momentum) {
        this.momentum = momentum;
        return this;
    }

    public double getL2RegularizationConstant() {
        return l2RegularizationConstant;
    }

    public BasicHeuristic setL2RegularizationConstant(double l2RegularizationConstant) {
        this.l2RegularizationConstant = l2RegularizationConstant;
        return this;
    }

    public boolean isDynamicKillingWeights() {
        return dynamicKillingWeights;
    }

    public BasicHeuristic setDynamicKillingWeights(boolean dynamicKillingWeights) {
        this.dynamicKillingWeights = dynamicKillingWeights;
        return this;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public BasicHeuristic setBatchSize(int batchSize) {
        this.batchSize = batchSize;
        return this;
    }

    @Override
    public String toString() {
        return "Heuristic{" +
                "name='" + name + '\'' +
                ", learningRate=" + learningRate +
                ", momentum=" + momentum +
                ", l2RegularizationConstant=" + l2RegularizationConstant +
                ", dynamicKillingWeights=" + dynamicKillingWeights +
                ", batchSize=" + batchSize +
                '}';
    }
}
