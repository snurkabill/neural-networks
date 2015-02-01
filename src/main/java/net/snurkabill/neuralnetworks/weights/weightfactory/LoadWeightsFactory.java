package net.snurkabill.neuralnetworks.weights.weightfactory;

import net.snurkabill.neuralnetworks.neuralnetwork.deep.DeepBoltzmannMachine;

import java.io.File;
import java.util.List;

public class LoadWeightsFactory implements WeightsFactory {

    private double weights[][][];

    public LoadWeightsFactory(File file) {
        //TODO : load weights
        throw new UnsupportedOperationException("Not implemented yet");
    }

    public LoadWeightsFactory(List<double[][]> wInput) {
        this.weights = new double[wInput.size()][][];
        for (int i = 0; i < wInput.size(); i++) {
            this.weights[i] = wInput.get(i);
        }
    }

    public LoadWeightsFactory(double[][][] wInput) {
        this.weights = wInput;
    }

    @Override
    public void setWeights(double[][][] weights) {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                System.arraycopy(this.weights[i][j], 0, weights[i][j], 0, weights[i][j].length);
            }
        }
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
