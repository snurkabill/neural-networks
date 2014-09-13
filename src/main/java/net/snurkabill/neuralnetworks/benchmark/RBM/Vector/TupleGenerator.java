package net.snurkabill.neuralnetworks.benchmark.RBM.Vector;

import java.util.ArrayList;
import java.util.List;

public class TupleGenerator {

    public static List<Tuple> createTuples(int lenght) {
        List<Tuple> tuples = new ArrayList<>();
        recursiveGenerator(tuples, 0, new int[lenght]);
        return tuples;
    }

    public static void recursiveGenerator(List<Tuple> tail, int index, int[] array) {
        if(index == array.length) {
            tail.add(new Tuple(array));
            return;
        }
        array[index] = 0;
        recursiveGenerator(tail, index + 1, array);
        array[index] = 1;
        recursiveGenerator(tail, index + 1, array);
    }
}
