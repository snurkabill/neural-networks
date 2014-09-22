package net.snurkabill.neuralnetworks.energybasednetwork;

public class HeuristicParamsRBM {
	
	public double temperature;
	public int numOfTrainingIterations;
	public int constructiveDivergenceIndex;
	public double learningSpeed;
	public double momentum;
	
	public static HeuristicParamsRBM createBasicHeuristicParams() {
		HeuristicParamsRBM rbm = new HeuristicParamsRBM();
		
		rbm.temperature = 1.0;
		rbm.constructiveDivergenceIndex = 3;
		rbm.numOfTrainingIterations = 30;
		rbm.momentum = 0.5;
		rbm.learningSpeed = 0.1;
		return rbm;
	} 
	
}
