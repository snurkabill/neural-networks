package net.snurkabill.neuralnetworks.benchmark;

import java.io.IOException;
import java.util.List;
import net.snurkabill.neuralnetworks.feedforwardnetwork.FeedForwardNetworkOfflineManager;
import net.snurkabill.neuralnetworks.results.BasicTestResults;

public class FeedForwardNetowkrBatchBenchmarker extends FeedForwardNetworkBenchmarker {

	protected final int sizeOfMiniBatch;
	
	public FeedForwardNetowkrBatchBenchmarker(FeedForwardNetworkOfflineManager feedForwardNetworks, int numOfRuns, 
			int sizeOfTrainingBatch, int sizeOfMiniBatch) {
		super(feedForwardNetworks, numOfRuns, sizeOfTrainingBatch);
		this.sizeOfMiniBatch = sizeOfMiniBatch;
	}

	@Override
	public void benchmark() {
		int percentageOfTrainingSet = 20;
		feedForwardNetworks.pretrainInputNeurons(percentageOfTrainingSet);
		for (int i = 0; i < numOfRuns; i++) {
			feedForwardNetworks.trainNetwork(sizeOfTrainingBatch, sizeOfMiniBatch);
			List<BasicTestResults> tmp = feedForwardNetworks.testNetworkOnWholeTestingDataset();
			for (int j = 0; j < tmp.size(); j++) {
				summary.get(j).add(tmp.get(j));
			}
		}
		try {
			makeReport();
		} catch (IOException ex) {
			throw new RuntimeException("Making report went wrong: " + ex);
		}
	}
}
