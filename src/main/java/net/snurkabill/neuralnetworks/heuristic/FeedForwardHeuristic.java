package net.snurkabill.neuralnetworks.heuristic;

import com.thoughtworks.xstream.annotations.XStreamAlias;

@XStreamAlias("FeedForwardHeuristic")
public class FeedForwardHeuristic extends BasicHeuristic {

    // TODO: remove dynamic boosting eta .. it seems counterproductive.
    public boolean dynamicBoostingEtas;
    public double BOOSTING_ETA_COEFF;
    public double KILLING_ETA_COEFF;

    public FeedForwardHeuristic() {
        super();
        this.dynamicBoostingEtas = false;
        this.BOOSTING_ETA_COEFF = 0;
        this.KILLING_ETA_COEFF = 0;
    }

    public boolean isDynamicBoostingEtas() {
        return dynamicBoostingEtas;
    }

    public FeedForwardHeuristic setDynamicBoostingEtas(boolean dynamicBoostingEtas) {
        this.dynamicBoostingEtas = dynamicBoostingEtas;
        return this;
    }

    public double getBOOSTING_ETA_COEFF() {
        return BOOSTING_ETA_COEFF;
    }

    public FeedForwardHeuristic setBOOSTING_ETA_COEFF(double BOOSTING_ETA_COEFF) {
        this.BOOSTING_ETA_COEFF = BOOSTING_ETA_COEFF;
        return this;
    }

    public double getKILLING_ETA_COEFF() {
        return KILLING_ETA_COEFF;
    }

    public FeedForwardHeuristic setKILLING_ETA_COEFF(double KILLING_ETA_COEFF) {
        this.KILLING_ETA_COEFF = KILLING_ETA_COEFF;
        return this;
    }

    @Override
    public String toString() {
        return "FeedForwardHeuristic { " +
                "name='" + this.getName() + '\'' +
                ", learningRate=" + learningRate +
                ", momentum=" + momentum +
                ", l2RegularizationConstant=" + l2RegularizationConstant +
                ", dynamicKillingWeights=" + dynamicKillingWeights +
                ", batchSize=" + batchSize +
                ", dynamicBoostingEtas=" + dynamicBoostingEtas +
                ", BOOSTING_ETA_COEFF=" + BOOSTING_ETA_COEFF +
                ", KILLING_ETA_COEFF=" + KILLING_ETA_COEFF +
                " }";
    }

    public static FeedForwardHeuristic createDefaultHeuristic() {
        FeedForwardHeuristic heuristic = new FeedForwardHeuristic();
        heuristic.learningRate = 0.001;
        heuristic.momentum = 0.1;
        heuristic.BOOSTING_ETA_COEFF = 1.000001;
        heuristic.KILLING_ETA_COEFF = 0.9999999;
        heuristic.l2RegularizationConstant = 0.999999;
        heuristic.dynamicBoostingEtas = false;
        heuristic.dynamicKillingWeights = false;
        return heuristic;
    }

    public static FeedForwardHeuristic deepProgressive() {
        FeedForwardHeuristic heuristic = new FeedForwardHeuristic();
        heuristic.learningRate = 0.01;
        heuristic.momentum = 0.1;
        heuristic.BOOSTING_ETA_COEFF = 1.0000001;
        heuristic.KILLING_ETA_COEFF = 0.99999999;
        heuristic.l2RegularizationConstant = 0.999999;
        heuristic.dynamicBoostingEtas = true;
        heuristic.dynamicKillingWeights = true;
        return heuristic;
    }

    public static FeedForwardHeuristic littleStepsWithHugeMoment() {
        FeedForwardHeuristic heuristic = new FeedForwardHeuristic();
        heuristic.learningRate = 0.0001;
        heuristic.momentum = 0.5;
        heuristic.BOOSTING_ETA_COEFF = 1.0000001;
        heuristic.KILLING_ETA_COEFF = 0.99999999;
        heuristic.l2RegularizationConstant = 0.999999;
        heuristic.dynamicBoostingEtas = true;
        heuristic.dynamicKillingWeights = false;
        return heuristic;
    }
}
