package net.snurkabill.neuralnetworks.math.linear;

public class Vector {

    public static void checkLength(Vector vector1, Vector vector2) {
        if(vector1.length != vector2.length) {
            throw new IllegalArgumentException("Vectors doesn't have the same length!");
        }
    }

    public static Vector copy(Vector vector) {
        return new Vector(vector.data);
    }

    public static void add(Vector vector, Vector other) {
        checkLength(vector, other);
        for (int i = 0; i < vector.length; i++) {
            vector.data[i] += other.data[i];
        }
    }

    public static void subtract(Vector vector, Vector other) {
        checkLength(vector, other);
        for (int i = 0; i < vector.length; i++) {
            vector.data[i] -= other.data[i];
        }
    }

    public static void multiply(Vector vector, Vector other) {
        checkLength(vector, other);
        for (int i = 0; i < vector.length; i++) {
            vector.data[i] *= other.data[i];
        }
    }

    public static void divide(Vector vector, Vector other) {
        checkLength(vector, other);
        for (int i = 0; i < vector.length; i++) {
            vector.data[i] /= other.data[i];
        }
    }

    public static void power(Vector vector, Vector other) {
        checkLength(vector, other);
        for (int i = 0; i < vector.length; i++) {
            vector.data[i] = Math.pow(vector.data[i], other.data[i]);
        }
    }

    public static void root(Vector vector, Vector other) {
        checkLength(vector, other);
        for (int i = 0; i < vector.length; i++) {
            vector.data[i] = Math.pow(vector.data[i], -other.data[i]);
        }
    }

    public static void add(Vector vector, double x) {
        for (int i = 0; i < vector.length; i++) {
            vector.data[i] += x;
        }
    }

    public static void subtract(Vector vector, double x) {
        for (int i = 0; i < vector.length; i++) {
            vector.data[i] -= x;
        }
    }

    public static void multiply(Vector vector, double x) {
        for (int i = 0; i < vector.length; i++) {
            vector.data[i] *= x;
        }
    }

    public static void divide(Vector vector, double x) {
        for (int i = 0; i < vector.length; i++) {
            vector.data[i] /= x;
        }
    }

    public static void power(Vector vector, double x) {
        for (int i = 0; i < vector.length; i++) {
            vector.data[i] = Math.pow(vector.data[i], x);
        }
    }

    public static void root(Vector vector, double x) {
        for (int i = 0; i < vector.length; i++) {
            vector.data[i] = Math.pow(vector.data[i], -x);
        }
    }

    public static void scalarProduct(Vector vector, Vector other) {
        checkLength(vector, other);
        double sum = 0.0;
        for (int i = 0; i < vector.length; i++) {
            sum += vector.data[i] * other.data[i];
        }
    }

    public final double[] data;
    public final int length;

    public Vector(double[] data) {
        this.data = data;
        this.length = data.length;
    }

}
