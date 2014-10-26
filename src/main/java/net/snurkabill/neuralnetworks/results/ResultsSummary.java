package net.snurkabill.neuralnetworks.results;

import java.util.ArrayList;
import java.util.List;

public class ResultsSummary {

    private String name;
    private List<TestResults> results = new ArrayList<>();

    public ResultsSummary(final String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public List<TestResults> getResults() {
        return results;
    }

    public void setResults(List<TestResults> results) {
        this.results = results;
    }

    public void add(TestResults resutls) {
        this.results.add(resutls);
    }
}
