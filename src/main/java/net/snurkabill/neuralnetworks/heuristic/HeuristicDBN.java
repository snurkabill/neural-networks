package net.snurkabill.neuralnetworks.heuristic;

public class HeuristicDBN extends HeuristicRBM {

    public int numOfRunsOfNetworkChimney;
    public int numofRunsBaseRBMItself;

    public HeuristicDBN() {
        super();
        this.numOfRunsOfNetworkChimney = 1;
        this.numofRunsBaseRBMItself = 1;
    }
}
