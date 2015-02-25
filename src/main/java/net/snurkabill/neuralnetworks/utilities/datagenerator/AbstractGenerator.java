package net.snurkabill.neuralnetworks.utilities.datagenerator;

import java.util.Random;

public abstract class AbstractGenerator {

    public static Random globalRandom = new Random(0);

    public abstract double generateNext();
}
