package net.snurkabill.neuralnetworks.feedforwardnetwork;

import net.snurkabill.neuralnetworks.feedforwardnetwork.heuristic.HeuristicParamsFFNN;
import net.snurkabill.neuralnetworks.feedforwardnetwork.weightfactory.WeightsFactory;
import net.snurkabill.neuralnetworks.utilities.Utilities;
import java.util.List;
import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FastFeedForwardNeuralNetwork {
	
	private static final Logger LOGGER = LoggerFactory.getLogger("FFNN");
	
	private final TransferFunctionCalculator transferFunction;
	private final Random random = new Random();
	private final List<Integer> topology;
	private final int lastLayerIndex;
	private final int sizeOfInputVector;
	private final int sizeOfOutputVector;
	private final int numOfLayers;
	
	private final double[] inputMeans;
	private final double[] inputStdDev;
	
	private final double[][] outputValues;
	private final double[][] gradients;
	
	private final double[][][] weights;
	private final double[][][] deltaWeights;
	private final double[][][] weightsEta;
	
	private HeuristicParamsFFNN heuristicParams;
	
	private double currentError;
	
	public FastFeedForwardNeuralNetwork(List<Integer> topology, WeightsFactory wFactory, 
			HeuristicParamsFFNN heuristicParams, TransferFunctionCalculator transferFunction, long seed) {
		this.random.setSeed(seed);
		this.transferFunction = transferFunction;
		this.topology = topology;
		this.heuristicParams = heuristicParams;
		this.lastLayerIndex = topology.size() - 1;
		this.sizeOfInputVector = topology.get(0);
		this.sizeOfOutputVector = topology.get(topology.size() - 1);
		this.numOfLayers = topology.size();
		this.outputValues = new double[topology.size()][];
		this.gradients = new double[topology.size()][];
		for (int i = 0; i < topology.size(); i++) {
			this.outputValues[i] = new double[topology.get(i) + 1];
			this.gradients[i] = new double[topology.get(i) + 1];
		}
		for (int i = 0; i < topology.size(); i++) {
			for (int j = 0; j < this.outputValues[i].length; j++) {
				this.outputValues[i][j] = 1; // BIAS values
			}
		}
		this.weights = new double[topology.size() - 1][][];
		this.deltaWeights = new double[topology.size() - 1][][];
		this.weightsEta = new double[topology.size() - 1][][];
		for (int i = 0; i < topology.size() - 1; i++) {
			this.weights[i] = new double[this.outputValues[i].length][];
			this.deltaWeights[i] = new double[this.outputValues[i].length][];
			this.weightsEta[i] = new double[this.outputValues[i].length][];	
			for (int j = 0; j < outputValues[i].length; j++) {
				this.weights[i][j] = new double[topology.get(i + 1)];
				this.deltaWeights[i][j] = new double[topology.get(i + 1)];
				this.weightsEta[i][j] = new double[topology.get(i + 1)];
			}	
		}
		wFactory.setWeights(this.weights);
		this.setEta();
		this.inputMeans = new double [this.sizeOfInputVector];
		this.inputStdDev = new double [this.sizeOfInputVector];
		for (int i = 0; i < this.sizeOfInputVector; i++) {
			this.inputMeans[i] = 0.0;
			this.inputStdDev[i] = 1.0;
		}
		LOGGER.info("Created Feed forward neural network {}", topology.toString());
	}
	
	public void setWeights(int layer, double[][] weights) {
		this.weights[layer] = weights;
	}
	
	public int getSizeOfInputVector() {
		return sizeOfInputVector;
	}
	
	public int getSizeOfOutputVector() {
		return sizeOfOutputVector;
	}

	public TransferFunctionCalculator getTransferFunction() {
		return transferFunction;
	}
	
	public HeuristicParamsFFNN getHeuristicParams() {
		return heuristicParams;
	}

	public void setHeuristicParams(HeuristicParamsFFNN heuristicParams) {
		this.heuristicParams = heuristicParams;
		setEta();
	}
	
	public void getResults(double[] results) {
		if(results.length != sizeOfOutputVector) {
			throw new IllegalArgumentException();
		}
		System.arraycopy(outputValues[lastLayerIndex], 0, results, 0, results.length);
	}
	
	public double getError(double[] targetValues) {
		double sum = 0.0;
		for (int i = 0; i < topology.get(lastLayerIndex); i++) {
			double tmp = outputValues[lastLayerIndex][i] - targetValues[i];
			sum += (tmp * tmp);
		}
		return (0.5) * sum;
	}
	
	private void setEta() {
		LOGGER.debug("Setting eta to each weight");
		for (int i = 0; i < topology.size() - 1; i++) {
			for (int j = 0; j < outputValues[i].length; j++) {
				for (int k = 0; k < topology.get(i + 1); k++) {
					weightsEta[i][j][k] = heuristicParams.eta;
				}
			}
		}
	}
	
	public int getFirstHighestValueIndex() {
		double max = this.transferFunction.getLowLimit(); 
		int bestIndex = -1;
		for (int i = 0; i < sizeOfOutputVector; i++) {
			if(outputValues[lastLayerIndex][i] > max) { 
				max = outputValues[lastLayerIndex][i]; 
				bestIndex = i;
			}
		}
		return bestIndex;
	}
	
	public int getLastHighestValueIndex() {
		double max = this.transferFunction.getLowLimit();
		int bestIndex = -1;
		for (int i = 0; i < sizeOfOutputVector; i++) {
			if(outputValues[lastLayerIndex][i] >= max) { 
				max = outputValues[lastLayerIndex][i]; 
				bestIndex = i;
			}
		}
		return bestIndex;
	}

	public void shiftWeights(double scale) {
		LOGGER.debug("Shifting weights");
		for (double[][] weight : weights) {
			for (double[] weight1 : weight) {
				for (int k = 0; k < weight1.length; k++) {
					weight1[k] += weight1[k] * scale * (random.nextGaussian() > 0 ? 1.0 : -1.0); 
				}
			}
		}
	}

	public void updateInputModifier(int index, double [] batch) {
		inputMeans[index] = Utilities.mean(batch);
		inputStdDev[index] = Utilities.stddev(batch, inputMeans[index]);
		if(inputStdDev[index] == 0.0) {
			inputStdDev[index] = 1.0;
		}
		LOGGER.trace("Index: {}, Mean: {}, StDev: {}", index, inputMeans[index], inputStdDev[index]);
	}
	// ********************************************************************************
	// feedForward methods
	// ********************************************************************************
	public void feedForwardWithPretrainedInputs(double[] inputVector) {
		if(inputVector.length != sizeOfInputVector) {
			throw new IllegalArgumentException();
		}
		for (int i = 0; i < sizeOfInputVector; i++) {
			outputValues[0][i] = (inputVector[i] - inputMeans[i]) / inputStdDev[i];
		}
		feedForward();
	}
	
	public void simpleFeedForward(double[] inputVector) {
		if(inputVector.length != sizeOfInputVector) {
			throw new IllegalArgumentException();
		}
		System.arraycopy(inputVector, 0, outputValues[0], 0, sizeOfInputVector);
		feedForward();
	}
	
	private void feedForward() {
		for (int i = 1; i < numOfLayers; i++) {
			for (int j = 0; j < topology.get(i); j++) {
				double sumOfInputs = 0.0;
				for (int k = 0; k < outputValues[i - 1].length; k++) {
					sumOfInputs += outputValues[i - 1][k] * weights[i - 1][k][j];
				}
				outputValues[i][j] = transferFunction.calculateOutputValue(sumOfInputs);
			}
		}
	}
	// ********************************************************************************
	// back propagation methods
	// ********************************************************************************
	public double backPropagation(double[] targetValues) {
		
		calcOutputGradients(targetValues);
		calcHiddenGradients();
		
		if(heuristicParams.dynamicKillingWeights) {
			if(heuristicParams.dynamicBoostingEtas) {
				complexUpdateWeights();
			} else {
				weightsKillingUpdateWeights();
			}
		} else if(heuristicParams.dynamicBoostingEtas) {
			etaBoostUpdateWeights();
		} else {
			simpleUpdateWeights();
		}
		return currentError;
	}
	
	private void calcOutputGradients(double[] targetValues) {
		if(targetValues.length != sizeOfOutputVector) {
			throw new IllegalArgumentException();
		}
		double error;
		currentError = 0.0;
		for (int i = 0; i < topology.get(lastLayerIndex) ; i++) {
			error = (targetValues[i] - outputValues[lastLayerIndex][i]);
			gradients[lastLayerIndex][i] = error *transferFunction.calculateDerivative(outputValues[lastLayerIndex][i]);
			currentError += error * error;
		}
		currentError *= 0.5;
	}
	
	private void calcHiddenGradients() {
		for (int i = numOfLayers - 2; i > 0; i--) {	
			for (int j = 0; j < topology.get(i); j++) {
				double sum = 0.0;
				for (int k = 0; k < topology.get(i + 1); k++) {
					sum += weights[i][j][k] * gradients[i + 1][k];
				}
				gradients[i][j] = sum * transferFunction.calculateDerivative(outputValues[i][j]);
			}
		}
	}
	
	// ********************************************************************************
	// methods for updating weights
	// ********************************************************************************
	private double calcDeltaWeight(int i, int j, int k) {
		double oldDeltaWeight = deltaWeights[i][k][j];
		deltaWeights[i][k][j] = weightsEta[i][k][j] * outputValues[i][k] * gradients[i + 1][j] 
							+ heuristicParams.alpha * oldDeltaWeight;
		return oldDeltaWeight;
	}
	
	private void simpleUpdateWeights() {
		for (int i = numOfLayers - 1; i > 0; i--) {
			for (int j = 0; j < topology.get(i); j++) {
				for (int k = 0; k < outputValues[i - 1].length; k++) {
					calcDeltaWeight(i - 1, j, k);
					weights[i - 1][k][j] += deltaWeights[i - 1][k][j];	
				}
			}
		}
	}
	
	private void complexUpdateWeights() {
		for (int i = numOfLayers - 1; i > 0; i--) {
			for (int j = 0; j < topology.get(i); j++) {
				for (int k = 0; k < outputValues[i - 1].length; k++) {
					double oldDeltaWeight = calcDeltaWeight(i - 1, j, k);
					if(oldDeltaWeight * deltaWeights[i - 1][k][j] >= 0) {
						 weightsEta[i - 1][k][j] *= heuristicParams.BOOSTING_ETA_COEFF;
					} else {
						 weightsEta[i - 1][k][j] *= heuristicParams.KILLING_ETA_COEFF;
					}
					weights[i - 1][k][j] =  heuristicParams.WEIGHT_DECAY * 
							(weights[i - 1][k][j] + deltaWeights[i - 1][k][j]);	
				}
			}
		}
	}
	
	private void etaBoostUpdateWeights() {
		for (int i = numOfLayers - 1; i > 0; i--) {
			for (int j = 0; j < topology.get(i); j++) {
				for (int k = 0; k < outputValues[i - 1].length; k++) {
					double oldDeltaWeight = deltaWeights[i - 1][k][j];
					deltaWeights[i - 1][k][j] = weightsEta[i - 1][k][j] * outputValues[i - 1][k] * gradients[i][j] 
							+ heuristicParams.alpha * oldDeltaWeight;
					if(oldDeltaWeight * deltaWeights[i - 1][k][j] >= 0) {
						 weightsEta[i - 1][k][j] *= heuristicParams.BOOSTING_ETA_COEFF;
					} else {
						 weightsEta[i - 1][k][j] *= heuristicParams.KILLING_ETA_COEFF;
					}
					weights[i - 1][k][j] += deltaWeights[i - 1][k][j];	
				}
			}
		}
	}
	
	private void weightsKillingUpdateWeights() {
		for (int i = numOfLayers - 1; i > 0; i--) {
			for (int j = 0; j < topology.get(i); j++) {
				for (int k = 0; k < outputValues[i - 1].length; k++) {
					calcDeltaWeight(i - 1, j, k);
					weights[i - 1][k][j] =  heuristicParams.WEIGHT_DECAY * 
							(weights[i - 1][k][j] + deltaWeights[i - 1][k][j]);	
				}
			}
		}
	}
	
	//TODO : implements mini-batch training
}
