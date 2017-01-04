package neural_network;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

import neural_network.gesture.DecimalPoint;
import static neural_network.NeuralNetworkUI.alpha;
import static neural_network.NeuralNetworkUI.beta;
import static neural_network.NeuralNetworkUI.gamma;
import static neural_network.NeuralNetworkUI.delta;
import static neural_network.NeuralNetworkUI.epsilon;

public class NeuralNetwork {
    // learning rate
    private double lr;
    // minimum error
    private double minE;
    // network architecture
    private Layer[] layers;
    // number of points in gesture representation, determines size of input
    // layer (2*M)
    int M;
    // sample's inputs
    private ArrayList<double[]> samplesX;
    // sample's outputs
    private ArrayList<double[]> samplesT;
    // 0 - backpropagation
    private int B = 0;
    // 1 - stohastic backpropagation
    private int SB = 1;
    // 2 - mini-batch propagation
    private int MBB = 2;
    private int batchSize = 5;
    private static NeuralNetwork nn;
    // is the network trained
    protected boolean trained;

    private NeuralNetwork() {
    }

    static NeuralNetwork getInstance() {
        if (nn == null) {
            return new NeuralNetwork();
        } else {
            return nn;
        }
    }

    public void train(String file, String m, String architecture,
            String learningRate, String minError, int algVer) {
        // parsing parameters
        M = Integer.parseInt(m);
        lr = Double.parseDouble(learningRate);
        minE = Double.parseDouble(minError);

        try {
            initializeNN(architecture);
        } catch (NeuralNetworkException e1) {
            System.out
                    .println(e1.getMessage()
                            + "\nNetwork training failed during network initialization.");
            return;
        }
        loadSamples(file);

        // training
        while (true) {
            double[] errors = new double[samplesX.size()];
            for (int sampleIndex = 0; sampleIndex < samplesX.size(); sampleIndex++) {
                // assign sample to the network input
                double[] y = samplesX.get(sampleIndex);
                // forward pass
                double[] o = forwardPassNN(y);
                // expected output
                double[] t = samplesT.get(sampleIndex);
                // compute and store error for current sample
                double e = 0;
                for (int i = 0; i < o.length; i++) {
                    e += Math.pow(t[i] - o[i], 2);
                }
                errors[sampleIndex] = e;

                // backward pass
                // computing errors for neurons of the last layer
                Layer outputLayer = layers[layers.length - 1];
                for (Neuron n : outputLayer) {
                    n.error = o[n.i] * (1 - o[n.i]) * (t[n.i] - o[n.i]);
                }

                // computing errors for the rest of the network
                for (int k = layers.length - 2; k > 0; k--) {
                    for (Neuron n : layers[k]) {
                        double s = weightedError(n);
                        n.error = n.y * (1 - n.y) * s;
                    }
                }
                // correct or accumulate weights
                correctOrAccumulateWeights(algVer);

                // accumulate error if needed
                if ((algVer == B && isLastSample(sampleIndex))
                        || (algVer == MBB && batchFull(sampleIndex))) {
                    addAccumulatedError();
                }
            }
            // mean square error
            double E = E(errors);
            System.out.println(E(errors));
            // when minimum error is reached, training is done
            if (E < minE) {
                break;
            }
        }
        trained = true;
    }

    private boolean isLastSample(int sampleIndex) {
        return sampleIndex == samplesX.size() - 1;
    }

    private boolean batchFull(int sampleIndex) {
        return (sampleIndex + 1) % batchSize == 0;
    }

    private void addAccumulatedError() {
        for (int k = 0; k < layers.length - 1; k++) {
            for (Neuron n : layers[k]) {
                for (int j = 0; j < n.e.length; j++) {
                    n.w[j] += n.e[j];
                    n.e[j] = 0;
                }
            }
        }
    }

    private void correctOrAccumulateWeights(int algorithmVersion) {
        for (int k = 0; k < layers.length - 1; k++) {
            for (Neuron n : layers[k]) {
                for (int j = 0; j < n.w.length; j++) {
                    double error = layers[k + 1].neurons[j].error;
                    if (algorithmVersion == SB) {
                        // correct weights if algorithm is stohastic
                        // backpropagation
                        n.w[j] += lr * n.y * error;
                    } else {
                        n.e[j] += lr * n.y * error;
                    }
                }
            }
        }
    }

    private double E(double[] errors) {
        double E = 0;
        for (double e : errors) {
            E += e;
        }
        return E / (2 * errors.length);
    }

    // weighted error of neurons to which neuron n sends its output
    private double weightedError(Neuron n) {
        int k = n.k;
        double sum = 0;
        for (int j = 0; j < layers[k + 1].size; j++) {
            double e = layers[k + 1].neurons[j].error;
            sum += e * n.w[j];
        }
        return sum;
    }

    private double[] forwardPassNN(double[] input) {
        int k = 0;
        // forward pass input layer
        Layer inputLayer = layers[k];
        for (Neuron n : inputLayer) {
            n.y = input[n.i];
        }
        for (k = 1; k < layers.length; k++) {
            forwardPassLayer(k);
        }
        Layer outputLayer = layers[k - 1];
        return outputLayer.output();
    }

    // pass k-th layer
    private void forwardPassLayer(int k) {
        for (Neuron n : layers[k]) {
            double net = net(n);
            n.y = Neuron.sigm(net);
        }
    }

    // net
    private double net(Neuron n) {
        double net = 0;
        Neuron[] neurons = layers[n.k - 1].neurons;
        for (Neuron neuron : neurons) {
            net += neuron.y * neuron.w[n.i];
        }
        return net;
    }

    /**
     * Initializes network with the given architecture, sets all weights to
     * random values.
     * 
     * @param architecture
     *            Defines network's architecture k1xk2x...xkn, k1 is size of the
     *            input layer, kn is size of the output layer.
     * @throws NeuralNetworkException
     *             If input layer's size is not 2*M where M is the number of
     *             points in the gesture representation.
     */
    private void initializeNN(String architecture)
            throws NeuralNetworkException {
        String[] arch = architecture.split("x");

        // creating layers of neural network
        layers = new Layer[arch.length];
        for (int layerIndex = 0; layerIndex < arch.length; layerIndex++) {
            int layerSize = Integer.parseInt(arch[layerIndex]);
            if (layerIndex == 0) {
                if (layerSize != 2 * M) {
                    throw new NeuralNetworkException(
                            "Size of the input layer should be 2*M.");
                }
            }
            layers[layerIndex] = new Layer(layerSize);
        }

        Random random = new Random();
        // go through layers
        for (int k = 0; k < layers.length; k++) {
            // for layer create neurons
            for (int j = 0; j < layers[k].size; j++) {
                Neuron neuron = new Neuron();
                neuron.k = k;
                neuron.i = j;
                // output layer doesn't have weights
                if (k < layers.length - 1) {
                    double[] w = new double[layers[k + 1].size];
                    double[] e = new double[layers[k + 1].size];
                    for (int i = 0; i < w.length; i++) {
                        w[i] = random.nextDouble();
                        e[i] = 0;
                    }
                    neuron.e = e;
                    neuron.w = w;
                }
                layers[k].neurons[j] = neuron;
            }
        }
    }

    /**
     * Recognizes gesture represented with decimal points. Works properly only
     * when {@link NeuralNetwork#trained} is set to true.
     * 
     * @param g
     *            Gesture representation
     * @return String that says which gesture was passed as parameter. Possible
     *         Strings are: alpha, beta, gamma, delta, epsilon
     * @throws NeuralNetworkException
     */
    public String infer(ArrayList<DecimalPoint> g)
            throws NeuralNetworkException {
        if (!trained) {
            throw new NeuralNetworkException(
                    "Cannot infer, network is not trained.");
        }

        double[] x = new double[g.size() * 2];

        for (int i = 0; i < g.size(); i++) {
            x[2 * i] = g.get(i).x;
            x[2 * i + 1] = g.get(i).y;
        }

        double[] result = forwardPassNN(x);
        int maxIndex = 0;
        for (int i = 0; i < result.length; i++) {
            if (result[i] > result[maxIndex]) {
                maxIndex = i;
            }
        }

        // debug
        System.out.print("alpha beta gamma delta epsilon:");
        for (double r : result) {
            System.out.print(r + " ");
        }
        System.out.println();

        if (maxIndex == 0) {
            return alpha;
        }

        if (maxIndex == 1) {
            return beta;
        }

        if (maxIndex == 2) {
            return gamma;
        }

        if (maxIndex == 3) {
            return delta;
        }
        return epsilon;
    }

    private void loadSamples(String file) {
        try {
            samplesX = new ArrayList<double[]>();
            samplesT = new ArrayList<double[]>();
            Scanner sc = new Scanner(new File(file));
            while (sc.hasNextLine()) {
                double[] x = new double[2 * M];

                for (int i = 0; i < M; i++) {
                    x[2 * i] = sc.nextDouble();
                    x[2 * i + 1] = sc.nextDouble();
                }
                samplesX.add(x);
                int outputLayerSize = layers[layers.length - 1].size;
                double[] y = new double[outputLayerSize];
                for (int i = 0; i < outputLayerSize; i++) {
                    y[i] = sc.nextInt();
                }
                samplesT.add(y);

            }
            sc.close();
        } catch (FileNotFoundException e) {
            System.out.println("Can't find file " + file);
        }
    }
}
