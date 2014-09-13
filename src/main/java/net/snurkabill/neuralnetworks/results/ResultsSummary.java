package net.snurkabill.neuralnetworks.results;

import java.util.ArrayList;
import java.util.List;

public class ResultsSummary {
	
	private String name;
	private List<BasicTestResults> results = new ArrayList<>();

    public ResultsSummary(final String name) {
        this.name = name;
    }

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public List<BasicTestResults> getResults() {
		return results;
	}

	public void setResults(List<BasicTestResults> results) {
		this.results = results;
	}
	
	public void add(BasicTestResults resutls) {
		this.results.add(resutls);
	}
}
