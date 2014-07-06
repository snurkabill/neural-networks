package net.snurkabill.neuralnetworks.feedforwardnetwork;

import java.util.ArrayList;
import java.util.List;
import net.snurkabill.neuralnetworks.feedforwardnetwork.target.SeparableTargetValues;
import net.snurkabill.neuralnetworks.feedforwardnetwork.target.TargetValues;

public abstract class FeedForwardNetworkManager {
	
	protected final int sizeOfOutputVector;
	protected final int sizeOfInputVector;
	protected final List<FeedForwardNeuralNetwork> networks;
	// TODO: general evaluator
	protected final List<TargetValues> targetMaker;

	public FeedForwardNetworkManager(List<FeedForwardNeuralNetwork> networks) {
		this.networks = networks;
		this.targetMaker = new ArrayList<>();
		for (int i = 0; i < networks.size(); i++) {
			targetMaker.add(new SeparableTargetValues(networks.get(i).getTransferFunction()));
		}
		this.sizeOfInputVector = networks.get(0).getSizeOfInputVector();
		this.sizeOfOutputVector = networks.get(0).getSizeOfOutputVector();
		for (int i = 0; i < networks.size(); i++) {
			if(networks.get(i).getSizeOfInputVector() != this.sizeOfInputVector) {
				throw new IllegalArgumentException("Networks have different input vector sizes");
			}
			if(networks.get(i).getSizeOfOutputVector() != this.sizeOfOutputVector) {
				throw new IllegalArgumentException("Networks have different output vector sizes");
			}
		}
	}
	
	public List<FeedForwardNeuralNetwork> getNetworkList() {
		return this.networks;
	}
	
	protected double[] initializeTargetArray(int network) {
		double[] targetValues = new double [networks.get(network).getSizeOfOutputVector()];
		for (int i = 0; i < networks.get(network).getSizeOfOutputVector(); i++) {
			targetValues[i] = networks.get(network).getTransferFunction().getLowLimit();
		}
		return targetValues;
	}
	
	public int getNumOfNeuralNetworks() {
		return this.networks.size();
	}	
}
