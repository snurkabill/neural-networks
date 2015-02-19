package net.snurkabill.neuralnetworks.heuristic;

import com.thoughtworks.xstream.annotations.XStreamAlias;

@XStreamAlias("BoltzmannMachineHeuristic")
public class BoltzmannMachineHeuristic extends FeedForwardHeuristic {

    public double temperature;
    public int constructiveDivergenceIndex;

    public BoltzmannMachineHeuristic() {
        super();
        this.temperature = 1;
        this.constructiveDivergenceIndex = 1;
    }

    public double getTemperature() {
        return temperature;
    }

    public BoltzmannMachineHeuristic setTemperature(double temperature) {
        this.temperature = temperature;
        return this;
    }

    public int getConstructiveDivergenceIndex() {
        return constructiveDivergenceIndex;
    }

    public BoltzmannMachineHeuristic setConstructiveDivergenceIndex(int constructiveDivergenceIndex) {
        this.constructiveDivergenceIndex = constructiveDivergenceIndex;
        return this;
    }

    @Override
    public String toString() {
        return "BoltzmannMachineHeuristic { " +
                "name='" + this.getName() + '\'' +
                ", learningRate=" + learningRate +
                ", momentum=" + momentum +
                ", l2RegularizationConstant=" + l2RegularizationConstant +
                ", dynamicKillingWeights=" + dynamicKillingWeights +
                ", batchSize=" + batchSize +
                ", dynamicBoostingEtas=" + dynamicBoostingEtas +
                ", BOOSTING_ETA_COEFF=" + BOOSTING_ETA_COEFF +
                ", KILLING_ETA_COEFF=" + KILLING_ETA_COEFF +
                ", temperature=" + temperature +
                ", constructiveDivergenceIndex=" + constructiveDivergenceIndex +
                " }";
    }

    @Deprecated
    public static BoltzmannMachineHeuristic createBasicHeuristicParams() {
        BoltzmannMachineHeuristic rbm = new BoltzmannMachineHeuristic();
        rbm.temperature = 1;
        rbm.constructiveDivergenceIndex = 3;
        rbm.batchSize = 10;
        rbm.momentum = 0.3;
        rbm.learningRate = 0.01;
        return rbm;
    }

    public static BoltzmannMachineHeuristic createStartingHeuristicParams() {
        BoltzmannMachineHeuristic heuristic = new BoltzmannMachineHeuristic();
        heuristic.constructiveDivergenceIndex = 1;
        heuristic.batchSize = 1;
        heuristic.temperature = 1;
        heuristic.learningRate = 0.1;
        heuristic.momentum = 0.1;
        return heuristic;
    }

}
