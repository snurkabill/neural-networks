package net.snurkabill.neuralnetworks.energybasednetwork;

import net.snurkabill.neuralnetworks.feedforwardnetwork.weightfactory.WeightsFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class RestrictedBoltzmannMachine extends BoltzmannMachine {

	protected final Logger LOGGER;
	
	private static final int FIX_VISIBLE = 0;
	
	protected final int sizeOfVisibleVector;
	protected final int sizeOfHiddenVector;
	
	protected final double[] visibleNeurons;
	protected final double[] hiddenNeurons;
	protected final double[] visibleBias;
	protected final double[] hiddenBias;
	protected final double[][] weights;
	
	private HeuristicParamsRBM heuristicParams;

	public RestrictedBoltzmannMachine(int numOfVisible, int numOfHidden, WeightsFactory wFactory, 
			HeuristicParamsRBM heuristicParams, long seed) {
		super(seed);
		if(numOfVisible <= 0 || numOfHidden <= 0) {
			throw new IllegalArgumentException("Wrong size of RBM");
		}
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
		wFactory.setWeights(visibleBias);
		wFactory.setWeights(hiddenBias);
		LOGGER = LoggerFactory.getLogger("RBM " + numOfVisible + " -> " + numOfHidden);
		LOGGER.info("Created");
	}

	public HeuristicParamsRBM getHeuristicParams() {
		return heuristicParams;
	}

	public final void setHeuristicParams(HeuristicParamsRBM heuristicParams) {
		//TODO: check heuristic values
		this.heuristicParams = heuristicParams;
		super.temperature = heuristicParams.temperature;
	}

    public double[][] createLayerForFFNN() {
        double[][] weightsLayer = new double[this.sizeOfVisibleVector + 1][sizeOfHiddenVector];
        for (int i = 0; i < this.sizeOfVisibleVector; i++) {
            for (int j = 0; j < this.sizeOfHiddenVector; j++) {
                weightsLayer[i][j] = weights[i][j];
            }
        }
        for (int i = 0; i < sizeOfHiddenVector; i++) {
            weightsLayer[this.sizeOfVisibleVector][i] = this.hiddenBias[i];
        }
        return weightsLayer;
    }

	public double[] getVisibleNeurons() {
		return visibleNeurons;
	}

	public int getSizeOfVisibleVector() {
		return sizeOfVisibleVector;
	}
	
	public int getSizeOfHiddenVector() {
		return sizeOfHiddenVector;
	}
	
	public void setVisibleNeurons(double[] visibleNeurons) {
		if(visibleNeurons.length != sizeOfVisibleVector) {
			throw new IllegalArgumentException("VisibleNeurons have different size!");
		}
		LOGGER.trace("Setting visible neurons");
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
		LOGGER.trace("Training machine started! InputVectorLength {}, sizeOfVisibleVector {}"
				, inputVals.length, sizeOfVisibleVector);
		if(inputVals.length != sizeOfVisibleVector) {
			throw new IllegalArgumentException("Wrong size of inputVector");
		}
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
	
	public void trainMachine() {
		this.trainMachine(visibleNeurons);
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
	
	/*public double[] reconstructNext() {
		machineStep();
		return this.getVisibleNeurons();
	}*/

    public abstract double[] reconstructNext();
	
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
	
	public double[] activateHiddenNeurons() {
		LOGGER.trace("Activating hidden neurons");
		calcHiddenNeurons();
		return this.getHiddenNeurons();
	}
	
	public double[] activateVisibleNeurons() {
		LOGGER.trace("Activating visible neurons");
		calcVisibleNeurons();
		return this.getVisibleNeurons();
	}
}
