package net.snurkabill.neuralnetworks.benchmark.FFNN;

import java.util.List;
import net.snurkabill.neuralnetworks.feedforwardnetwork.manager.FeedForwardNetworkOfflineManager;
import net.snurkabill.neuralnetworks.results.BasicTestResults;

public class FeedForwardNetworkBatchBenchmarker extends FeedForwardNetworkBenchmarker {

	protected final int sizeOfMiniBatch;
	
	public FeedForwardNetworkBatchBenchmarker(FeedForwardNetworkOfflineManager feedForwardNetworks, int numOfRuns,
                                              int sizeOfTrainingBatch, int sizeOfMiniBatch) {
		super(feedForwardNetworks, numOfRuns, sizeOfTrainingBatch);
		this.sizeOfMiniBatch = sizeOfMiniBatch;
	}

	@Override
	public void benchmark() {
		for (int i = 0; i < numOfRuns; i++) {
			feedForwardNetworks.trainNetwork(sizeOfTrainingBatch, sizeOfMiniBatch);
			List<BasicTestResults> tmp = feedForwardNetworks.testNetworkOnWholeTestingDataset();
			for (int j = 0; j < tmp.size(); j++) {
				summary.get(j).add(tmp.get(j));
			}
		}
        makeReport();
	}
}