package neural_network;

public class Neuron {
    // layer index
    protected int k;
    // index
    protected int i;
    // output
    protected double y;
    // weights
    protected double[] w;
    // error
    protected double error;
    // accumulated weight correction
    double[] e;

    // transfer function
    public static double sigm(double net) {
        return 1.0 / (1.0 + Math.pow(Math.E, -net));
    }
}
