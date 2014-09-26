package net.snurkabill.neuralnetworks.feedforwardnetwork.manager;

import java.util.ArrayList;
import java.util.List;

import net.snurkabill.neuralnetworks.feedforwardnetwork.core.FeedForwardNeuralNetwork;
import net.snurkabill.neuralnetworks.feedforwardnetwork.target.SeparableTargetValues;
import net.snurkabill.neuralnetworks.feedforwardnetwork.target.TargetValues;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class FeedForwardNetworkManager {

    protected static final Logger LOGGER = LoggerFactory.getLogger("FFNN Manager");

    protected final int sizeOfOutputVector;
	protected final int sizeOfInputVector;

    protected final List<FeedForwardNeuralNetwork> networks;

	// TODO: general evaluator
	protected final List<TargetValues> targetMaker;

    private final int determinedNumOfThreads;
    private final boolean multiThreadingEnabled;

    protected int trainingIterations;

	public FeedForwardNetworkManager(List<FeedForwardNeuralNetwork> networks, boolean threadsEnabled) {
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

        multiThreadingEnabled = threadsEnabled;
        if(threadsEnabled) {
            int numOfThreads = networks.size();
            int safeNumOfThreads = Runtime.getRuntime().availableProcessors() - 1;
            if(numOfThreads > safeNumOfThreads) {
                numOfThreads = safeNumOfThreads;
            }
            determinedNumOfThreads = numOfThreads;
        } else {
            determinedNumOfThreads = 1;
        }

		LOGGER.info("Manager created with {} networks", networks.size());
        LOGGER.info("Threads enabled = {}, determined number of threads = {}", multiThreadingEnabled,
                determinedNumOfThreads);
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

    public boolean isMultiThreadingEnabled() {
        return this.multiThreadingEnabled;
    }

    public int numberOfThreads() {
        return this.determinedNumOfThreads;
    }
}
