import java.util.List;
import java.util.ArrayList;
import java.util.HashSet;

public class CrossValidation {
    /*
     * Returns the k-fold cross validation score of classifier clf on training data.
     */
    public static double kFoldScore(Classifier clf, List<Instance> trainData, int k, int v) {
        int N = trainData.size();
        if(k<2 || k>trainData.size()){
            return 0.0;
        }
        double culmAcc = 0.0;
        for(int i = 0; i < k; i++){
            
            List<Instance> trainSet = new ArrayList<Instance>();
            List<Instance> testSet = new ArrayList<Instance>();
            for(int j = 0; j < N; j++){
                if(Math.floor((double)j/((double)N/(double)k))==i){
                    trainSet.add(trainData.get(j));
                }else{
                    testSet.add(trainData.get(j));
                }
            }
            HashSet<String> dict = new HashSet<String>();
            for(Instance inst: trainSet){
                for(String s: inst.words){
                    dict.add(s);
                }
            }
            clf.train(trainSet, v);

            int numOfCorrect = 0;
            for(Instance inst: testSet){
                if(clf.classify(inst.words).label == inst.label){
                    numOfCorrect++;
                }
            }
            culmAcc += (double)numOfCorrect/(double)testSet.size();

        }

        return culmAcc/k;
    }
}
