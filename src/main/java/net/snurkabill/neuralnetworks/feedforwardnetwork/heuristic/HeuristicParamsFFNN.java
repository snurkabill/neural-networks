package net.snurkabill.neuralnetworks.feedforwardnetwork.heuristic;

public class HeuristicParamsFFNN {
	
	public double eta;
	public double alpha;
	public double WEIGHT_DECAY;
	public boolean dynamicBoostingEtas;
	public boolean dynamicKillingWeights;
	public double BOOSTING_ETA_COEFF;
	public double KILLING_ETA_COEFF;
	
	public byte[] getAsBytes() {
		// TODO   .... need XML instead
		return null;
	}
	
	public static HeuristicParamsFFNN createDefaultHeuristic() {
		HeuristicParamsFFNN heuristic = new HeuristicParamsFFNN();
		heuristic.eta = 0.0001;
		heuristic.alpha = 0.1;
		heuristic.BOOSTING_ETA_COEFF = 1.000001;
		heuristic.KILLING_ETA_COEFF = 0.9999999;
		heuristic.WEIGHT_DECAY = 0.999999;
		heuristic.dynamicBoostingEtas = false;
		heuristic.dynamicKillingWeights = false;
		return heuristic;
	}
	
	public static HeuristicParamsFFNN deepProgressive() {
		HeuristicParamsFFNN heuristic = new HeuristicParamsFFNN();
		heuristic.eta = 0.01;
		heuristic.alpha = 0.1;
		heuristic.BOOSTING_ETA_COEFF = 1.0000001;
		heuristic.KILLING_ETA_COEFF = 0.99999999;
		heuristic.WEIGHT_DECAY = 0.999999;
		heuristic.dynamicBoostingEtas = true;
		heuristic.dynamicKillingWeights = true;
		return heuristic;
	}
	
	public static HeuristicParamsFFNN littleStepsWithBigMoment() {
		HeuristicParamsFFNN heuristic = new HeuristicParamsFFNN();
		heuristic.eta = 0.0001;
		heuristic.alpha = 0.5;
		heuristic.BOOSTING_ETA_COEFF = 1.0000001;
		heuristic.KILLING_ETA_COEFF = 0.99999999;
		heuristic.WEIGHT_DECAY = 0.999999;
		heuristic.dynamicBoostingEtas = true;
		heuristic.dynamicKillingWeights = false;
		return heuristic;
	}
}
