package net.snurkabill.neuralnetworks.benchmark.FFNN;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import net.snurkabill.neuralnetworks.benchmark.Benchmarker;
import net.snurkabill.neuralnetworks.benchmark.report.ReportMaker;
import net.snurkabill.neuralnetworks.feedforwardnetwork.manager.FeedForwardNetworkOfflineManager;
import net.snurkabill.neuralnetworks.results.BasicTestResults;
import net.snurkabill.neuralnetworks.results.ResultsSummary;

public class FeedForwardNetworkBenchmarker implements Benchmarker {

	FeedForwardNetworkOfflineManager feedForwardNetworks;
	
	protected final int numOfRuns;
	protected final int sizeOfTrainingBatch;
	protected final List<ResultsSummary> summary;
	
	public FeedForwardNetworkBenchmarker(FeedForwardNetworkOfflineManager feedForwardNetworks, int numOfRuns, 
			int sizeOfTrainingBatch) {
		this.feedForwardNetworks = feedForwardNetworks;
		this.numOfRuns = numOfRuns;
		this.sizeOfTrainingBatch = sizeOfTrainingBatch;
		summary = new ArrayList<>();
		
		for (int i = 0; i < feedForwardNetworks.getNumOfNeuralNetworks(); i++) {
			summary.add(new ResultsSummary(feedForwardNetworks.getNetworkList().get(i).getName()));
		}
	}
	
	@Override
	public void benchmark() {
		for (int i = 0; i < numOfRuns; i++) {
			feedForwardNetworks.trainNetwork(sizeOfTrainingBatch);
			List<BasicTestResults> tmp = feedForwardNetworks.testNetworkOnWholeTestingDataset();
			for (int j = 0; j < tmp.size(); j++) {
				summary.get(j).add(tmp.get(j));
			}
		}
        makeReport();
	}
	
	public List<ResultsSummary> getResults() {
		return this.summary;
	}

    protected void makeReport() {
        ReportMaker.chart(new File("results.png"), summary, "FFNN", "iterations", "success");
    }
}
