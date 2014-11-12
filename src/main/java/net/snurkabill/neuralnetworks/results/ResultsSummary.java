package net.snurkabill.neuralnetworks.results;

import java.util.ArrayList;
import java.util.List;

public class ResultsSummary {

    private String name;
    private List<NetworkResults> results = new ArrayList<>();

    public ResultsSummary(final String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public List<NetworkResults> getResults() {
        return results;
    }

    public void setResults(List<NetworkResults> results) {
        this.results = results;
    }

    public void add(NetworkResults resutls) {
        this.results.add(resutls);
    }
}
