package net.snurkabill.neuralnetworks.feedforwardnetwork.weightfactory;

import java.io.File;

public class PretrainedWeightsFactory implements WeightsFactory {

	private double weihgts[][][];
	
	public PretrainedWeightsFactory(File file) {
		//TODO : load weihgts
		
	}
	
	@Override
	public void setWeights(double[][][] weights) {
		weights = this.weihgts;
	}

	@Override
	public void setWeights(int layerIndex, double[][] weights) {
			throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
	}

	@Override
	public void setWeights(int layerIndex, int neuronInedx, double[] weights) {
			throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
	}

	@Override
	public void setWeights(double[][] weights) {
		throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
	}

	@Override
	public void setWeights(double[] weights) {
		throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
	}
}
