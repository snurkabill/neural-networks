package net.snurkabill.neuralnetworks.feedforwardnetwork.heuristic;

public abstract class BackPropagateHeuristic {
	
	private HeuristicParamsFFNN heuristicParams;

	public HeuristicParamsFFNN getHeuristicParams() {
		return heuristicParams;
	}

	public void setHeuristicParams(HeuristicParamsFFNN heuristicParams) {
		this.heuristicParams = heuristicParams;
	}
	
}
