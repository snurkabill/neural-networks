package net.snurkabill.neuralnetworks.feedforwardnetwork;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import net.snurkabill.neuralnetworks.feedforwardnetwork.heuristic.HeuristicParamsFFNN;
import net.snurkabill.neuralnetworks.feedforwardnetwork.weightfactory.WeightsFactory;
import net.snurkabill.neuralnetworks.utilities.Utilities;
import java.util.List;
import java.util.Random;
import net.snurkabill.neuralnetworks.feedforwardnetwork.weightfactory.PretrainedWeightsFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FeedForwardNeuralNetwork {
	
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
	
	private final String name;
	
	private HeuristicParamsFFNN heuristicParams;
	
	private double currentError;
	
	public FeedForwardNeuralNetwork(List<Integer> topology, WeightsFactory wFactory, String name,  
			HeuristicParamsFFNN heuristicParams, TransferFunctionCalculator transferFunction, long seed) {
		if(topology.size() <= 1) {
			throw new IllegalArgumentException("Wrong topology of FFNN");
		}
		this.name = name;
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
	
	public FeedForwardNeuralNetwork(List<Integer> topology, WeightsFactory wFactory, String name,  
			HeuristicParamsFFNN heuristicParams, TransferFunctionCalculator transferFunction, long seed, 
			double[] inputMeans, double[] inputStdDev) {
		this(topology, wFactory, name, heuristicParams, transferFunction, seed);
		for (int i = 0; i < inputStdDev.length; i++) {
			this.inputMeans[i] = inputMeans[i];
			this.inputStdDev[i] = inputStdDev[i];
		}
	}
	
	public void setWeights(WeightsFactory wFactory) {
		wFactory.setWeights(weights);
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
	
	public String getName() {
		return this.name;
	}
	
	public HeuristicParamsFFNN getHeuristicParams() {
		return heuristicParams;
	}

	public void setHeuristicParams(HeuristicParamsFFNN heuristicParams) {
		this.heuristicParams = heuristicParams;
		setEta();
	}

	public double getCurrentError() {
		return currentError;
	}

	public void setCurrentError(double currentError) {
		this.currentError = currentError;
	}
	
	public void getResults(double[] results) {
		if(results.length != sizeOfOutputVector) {
			throw new IllegalArgumentException();
		}
		System.arraycopy(outputValues[lastLayerIndex], 0, results, 0, results.length);
	}
	
	public double getError(double[] targetValues) {
		if(targetValues.length != sizeOfOutputVector) {
			throw new IllegalArgumentException();
		}
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
		if(index < 0 || index >= sizeOfInputVector) {
			throw new IllegalArgumentException("Wrong index: " + index);
		}
		inputMeans[index] = Utilities.mean(batch);
		inputStdDev[index] = Utilities.stddev(batch, inputMeans[index]);
		if(inputStdDev[index] == 0.0) {
			inputStdDev[index] = 1.0;
		}
		LOGGER.trace("Index: {}, Mean: {}, StDev: {}", index, inputMeans[index], inputStdDev[index]);
	}
	
	public void saveNetwork() throws FileNotFoundException, IOException {
		this.saveNetwork(new File(this.name + "_weights"));
	}
	
	public void saveNetwork(File baseFile) throws FileNotFoundException, IOException {
            LOGGER.info("Saving network");
		try(DataOutputStream os = new DataOutputStream(new FileOutputStream(baseFile))) {
			saveWeights(os);
			saveInputModdifiers(os);
		}
	}
	
	protected void saveWeights(DataOutputStream os) throws IOException {
		os.writeInt(topology.size());
		for (int i = 0; i < topology.size(); i++) {
			os.writeInt(this.outputValues[i].length);
		}
		for (int i = 0; i < topology.size() - 1; i++) {
			for (int j = 0; j < outputValues[i].length; j++) {
				for (int k = 0; k < topology.get(i + 1); k++) {
					os.writeDouble(weights[i][j][k]);
				}
			}
		}
	}
	
	protected void saveInputModdifiers(DataOutputStream os) throws IOException {
		for (int i = 0; i < topology.get(0); i++) {
			os.writeDouble(this.inputMeans[i]);
		}
		for (int i = 0; i < topology.get(0); i++) {
			os.writeDouble(this.inputStdDev[i]);
		}
	}
	
	// ********************************************************************************
	// feedForward methods
	// ********************************************************************************
	public void feedForwardWithPretrainedInputs(double[] inputVector) {
		if(inputVector.length != sizeOfInputVector) {
			throw new IllegalArgumentException("InputVector has different size than neural network first "
					+ "layer");
		}
		for (int i = 0; i < sizeOfInputVector; i++) {
			outputValues[0][i] = (inputVector[i] - inputMeans[i]) / inputStdDev[i];
			LOGGER.trace("Neuron [0][{}], output: {}", i, outputValues[0][i]);
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
		if(targetValues.length != sizeOfOutputVector) {
			throw new IllegalArgumentException();
		}
		
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
	
	public static FeedForwardNeuralNetwork buildNeuralNetwork(File fWeights, String name, 
			HeuristicParamsFFNN heuristicParams, TransferFunctionCalculator transferFunction, long seed
				) throws FileNotFoundException, IOException {
		try(DataInputStream is = new DataInputStream(new FileInputStream(fWeights))) {
			return readingFromStream(is, name, heuristicParams, transferFunction, seed);
		}
	}
	
	protected static FeedForwardNeuralNetwork readingFromStream(DataInputStream is, String name, 
			HeuristicParamsFFNN heuristicParams, TransferFunctionCalculator transferFunction, long seed
				) throws IOException {
		int layers = is.readInt();
		List<Integer> topology = new ArrayList<>();
		for (int i = 0; i < layers; i++) {
			topology.add(is.readInt() - 1);
		}
		double[][][] weights = initializeWeights(topology); 
		loadWeights(weights, is);
		WeightsFactory wFactory = new PretrainedWeightsFactory(weights);
		if(is.available() != 0) {
			double[] inputMeans = new double[topology.get(0)];
			double[] inputStdDev = new double[topology.get(0)];
			for (int i = 0; i < inputMeans.length; i++) {
				inputMeans[i] = is.readDouble();
			}
			for (int i = 0; i < inputStdDev.length; i++) {
				inputStdDev[i] = is.readDouble();
			}
			return new FeedForwardNeuralNetwork(topology, wFactory, name, heuristicParams, transferFunction, seed,
					inputMeans, inputStdDev);
		} else {
			return new FeedForwardNeuralNetwork(topology, wFactory, name, heuristicParams, transferFunction, seed);
		}
	}
	
	protected static double[][][] initializeWeights(List<Integer> topology) {
		double[][][] weights;
		weights = new double[topology.size() - 1][][];
		for (int i = 0; i < topology.size() - 1; i++) {
			weights[i] = new double[topology.get(i) + 1][];
			for (int j = 0; j < topology.get(i) + 1; j++) {
				weights[i][j] = new double[topology.get(i + 1)];
			}	
		}
		return weights;
	}
	
	protected static void loadWeights(double[][][] weights, DataInputStream is) throws IOException {
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[i].length; j++) {
				for (int k = 0; k < weights[i][j].length; k++) {
					weights[i][j][k] = is.readDouble();
				}
			}
		}
	}
}
