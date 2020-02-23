import java.util.Random;
import java.io.PrintStream;
import java.io.FileOutputStream;
import java.io.FileNotFoundException;

public class ShuffleData{
    // use the shuffle way according to CrossValidation.java in Meka tool
    private int[] data;

    public GetData(int capacity){
        data = new int[capacity];

        for (int i = 0; i < capacity; i++){
            data[i] = i;
        }
    }

    public void swap(int i, int j){
        if (i == j){
            return;
        }

        int t = data[i];
        data[i] = data[j];
        data[j] = t;
    }

    public void randomize(Random random){
        for (int j = data.length - 1; j > 0; j--){
            swap(j, random.nextInt(j + 1));
        }
    }

    public static void main(String[] args){
        Random rand = new Random(0);
        GetData dataset = new GetData(780);
        dataset.randomize(rand);

        for (int i = 0; i < dataset.data.length; i ++){
            System.out.println(dataset.data[i]);
        }
    }

}