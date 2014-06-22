package net.snurkabill.neuralnetworks.energybasednetwork;

import net.snurkabill.neuralnetworks.feedforwardnetwork.weightfactory.WeightsFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * NOT FINISHED YET!
 * @author snurkabill
 */

public abstract class RestrictedBolztmannMachine extends BoltzmannMachine {

	protected static final Logger LOGGER = LoggerFactory.getLogger("RBM");
	
	private static final int FIX_VISIBLE = 0;
	
	protected final int sizeOfVisibleVector;
	protected final int sizeOfHiddenVector;
	
	protected final double[] visibleNeurons;
	protected final double[] hiddenNeurons;
	protected final double[] visibleBias;
	protected final double[] hiddenBias;
	protected final double[][] weights;
	
	private HeuristicParamsRBM heuristicParams;

	public RestrictedBolztmannMachine(int numOfVisible, int numOfHidden, WeightsFactory wFactory, 
			HeuristicParamsRBM heuristicParams, long seed) {
		super(seed);
		this.setHeuristicParams(heuristicParams);
		this.sizeOfVisibleVector = numOfVisible;
		this.sizeOfHiddenVector = numOfHidden;
		this.visibleNeurons = new double[numOfVisible];
		this.visibleBias = new double[numOfVisible];
		this.hiddenNeurons = new double[numOfHidden];
		this.hiddenBias = new double[numOfHidden];
		weights = new double[numOfVisible][];
		for (int i = 0; i < numOfVisible; i++) {
			weights[i] = new double[numOfHidden];
		}
		wFactory.setWeights(weights);
		/*wFactory.setWeights(visibleBias);
		wFactory.setWeights(hiddenBias);	
		*/LOGGER.info("Created Restricted Boltzmann Machine {} -> {}", numOfVisible, numOfHidden);
	}

	public HeuristicParamsRBM getHeuristicParams() {
		return heuristicParams;
	}

	public void setHeuristicParams(HeuristicParamsRBM heuristicParams) {
		this.heuristicParams = heuristicParams;
		super.temperature = heuristicParams.temperature;
	}

	public double[] getVisibleNeurons() {
		return visibleNeurons;
	}

	public int getSizeOfVisibleVector() {
		return sizeOfVisibleVector;
	}
	
	public void setVisibleNeurons(double[] visibleNeurons) {
		if(visibleNeurons.length != sizeOfVisibleVector) {
			throw new IllegalArgumentException("VisibleNeurons have different size!");
		}
		System.arraycopy(visibleNeurons, 0, this.visibleNeurons, 0, sizeOfVisibleVector);
	}

	public double[] getHiddenNeurons() {
		return hiddenNeurons;
	}
			
	protected double calcHiddenPotential(int index) {
		double potential = 0.0;
		for (int j = 0; j < sizeOfVisibleVector; j++) {
			potential += weights[j][index] * visibleNeurons[j];
		}
		potential += hiddenBias[index];
		return potential;
	}
	
	protected double calcVisiblePotential(int index) {
		double potential = 0.0;
		for (int i = 0; i < sizeOfHiddenVector; i++) {
			potential += weights[index][i] * hiddenNeurons[i];
		}
		potential += visibleBias[index];
		return potential;
	}	
	
	protected abstract void calcVisibleNeurons();
	protected abstract void calcHiddenNeurons();
	
	public void trainMachine(double[] inputVals) {
		LOGGER.trace("Training machine started!");
		double[][] diffWeights = runMachine(FIX_VISIBLE, inputVals);
		double[][] reconWeights = runMachine(heuristicParams.constructiveDivergenceIndex, inputVals);
		for (int i = 0; i < sizeOfVisibleVector; i++) {
			for (int j = 0; j < sizeOfHiddenVector; j++) {
				diffWeights[i][j] -= reconWeights[i][j];
			}
		}
		for (int i = 0; i < sizeOfVisibleVector; i++) {
			for (int j = 0; j < sizeOfHiddenVector; j++) {
				weights[i][j] -= (heuristicParams.learningSpeed/super.temperature) * diffWeights[i][j];
			}
		}
	}
	
	private double[][] runMachine(int iterations, double[] inputVals) {
		LOGGER.trace("Machine runs {} iterations", iterations);
		double[][] weightExpected = new double[sizeOfVisibleVector][sizeOfHiddenVector];
		for (int i = 0; i < heuristicParams.numOfTrainingIterations; i++) {
			constrastiveDivergence(iterations, inputVals);
			for (int j = 0; j < sizeOfVisibleVector; j++) {
				for (int k = 0; k < sizeOfHiddenVector; k++) {
					weightExpected[j][k] += visibleNeurons[j] * hiddenNeurons[k];
				}
				
			}
		}
		for (int i = 0; i < sizeOfVisibleVector; i++) {
			for (int j = 0; j < sizeOfHiddenVector; j++) {
				weightExpected[i][j] /= heuristicParams.numOfTrainingIterations;
			}
		}
		return weightExpected;
	}
	
	public double[] reconstructNext() {
		machineStep();
		return this.getVisibleNeurons();
	}
	
	private void constrastiveDivergence(int index, double[] inputVals) {
		setVisibleNeurons(inputVals);
		calcHiddenNeurons();
		for (int i = 0; i < index; i++) {
			machineStep();
		}
	}
	
	private void machineStep() {
		calcVisibleNeurons();
		calcHiddenNeurons();
	}
}
