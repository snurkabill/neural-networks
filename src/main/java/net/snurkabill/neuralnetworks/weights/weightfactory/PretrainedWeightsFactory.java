package net.snurkabill.neuralnetworks.weights.weightfactory;

import java.io.File;
import java.util.List;

public class PretrainedWeightsFactory implements WeightsFactory {

	private double weights[][][];
	
	public PretrainedWeightsFactory(File file) {
		//TODO : load weights
	}

    public PretrainedWeightsFactory(List<double[][]> wInput) {
        this.weights = new double[wInput.size()][][];
        for (int i = 0; i < wInput.size(); i++) {
            this.weights[i] = wInput.get(i);
        }
    }

	public PretrainedWeightsFactory(double[][][] wInput) {
		this.weights = wInput;
	}
	
	@Override
	public void setWeights(double[][][] weights) {
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[i].length; j++) {
				for (int k = 0; k < weights[i][j].length; k++) {
					weights[i][j][k] = this.weights[i][j][k];
				}
			}
		}
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
