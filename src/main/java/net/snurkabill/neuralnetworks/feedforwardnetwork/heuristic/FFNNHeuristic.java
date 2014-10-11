package net.snurkabill.neuralnetworks.feedforwardnetwork.heuristic;

import net.snurkabill.neuralnetworks.Heuristic;

public class FFNNHeuristic extends Heuristic {

	public double WEIGHT_DECAY;
	public boolean dynamicBoostingEtas;
	public boolean dynamicKillingWeights;
	public double BOOSTING_ETA_COEFF;
	public double KILLING_ETA_COEFF;
	
	public static FFNNHeuristic createDefaultHeuristic() {
		FFNNHeuristic heuristic = new FFNNHeuristic();
		heuristic.learningRate = 0.001;
		heuristic.momentum = 0.1;
		heuristic.BOOSTING_ETA_COEFF = 1.000001;
		heuristic.KILLING_ETA_COEFF = 0.9999999;
		heuristic.WEIGHT_DECAY = 0.999999;
		heuristic.dynamicBoostingEtas = false;
		heuristic.dynamicKillingWeights = false;
		return heuristic;
	}
	
	public static FFNNHeuristic deepProgressive() {
		FFNNHeuristic heuristic = new FFNNHeuristic();
		heuristic.learningRate = 0.01;
		heuristic.momentum = 0.1;
		heuristic.BOOSTING_ETA_COEFF = 1.0000001;
		heuristic.KILLING_ETA_COEFF = 0.99999999;
		heuristic.WEIGHT_DECAY = 0.999999;
		heuristic.dynamicBoostingEtas = true;
		heuristic.dynamicKillingWeights = true;
		return heuristic;
	}
	
	public static FFNNHeuristic littleStepsWithHugeMoment() {
		FFNNHeuristic heuristic = new FFNNHeuristic();
		heuristic.learningRate = 0.0001;
		heuristic.momentum = 0.5;
		heuristic.BOOSTING_ETA_COEFF = 1.0000001;
		heuristic.KILLING_ETA_COEFF = 0.99999999;
		heuristic.WEIGHT_DECAY = 0.999999;
		heuristic.dynamicBoostingEtas = true;
		heuristic.dynamicKillingWeights = false;
		return heuristic;
	}
}
