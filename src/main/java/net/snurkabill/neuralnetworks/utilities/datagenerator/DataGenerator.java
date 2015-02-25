package net.snurkabill.neuralnetworks.utilities.datagenerator;

import net.snurkabill.neuralnetworks.data.database.DataItem;

import java.util.List;

public class DataGenerator {

    private final List<NGaussianDistribution> distributions;

    public DataGenerator(List<NGaussianDistribution> generators) {
        this.distributions = generators;
    }

    public DataItem generateNext() {
        return distributions.get(AbstractGenerator.globalRandom.nextInt(distributions.size())).generateNextVector();
    }
}
