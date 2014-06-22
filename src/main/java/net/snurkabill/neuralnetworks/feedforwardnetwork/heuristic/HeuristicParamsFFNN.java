package net.snurkabill.neuralnetworks.feedforwardnetwork.heuristic;

public class HeuristicParamsFFNN {
	
	public double eta;
	public double alpha;
	public double WEIGHT_DECAY;
	public boolean dynamicBoostingEtas;
	public boolean dynamicKillingWeights;
	public double BOOSTING_ETA_COEFF;
	public double KILLING_ETA_COEFF;
	
	public static HeuristicParamsFFNN createDefaultHeuristic() {
		HeuristicParamsFFNN heuristic = new HeuristicParamsFFNN();
		heuristic.eta = 0.001;
		heuristic.alpha = 0.1;
		heuristic.BOOSTING_ETA_COEFF = 1.000001;
		heuristic.KILLING_ETA_COEFF = 0.9999999;
		heuristic.WEIGHT_DECAY = 0.999999;
		heuristic.dynamicBoostingEtas = false;
		heuristic.dynamicKillingWeights = false;
		return heuristic;
	}
}
