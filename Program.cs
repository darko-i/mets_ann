using Accord.IO;
using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.DecisionTrees.Learning;
using Accord.Math.Optimization.Losses;
using Accord.Neuro;
using Accord.Neuro.Learning;
using Accord.Statistics;
using Accord.Statistics.Models.Regression.Fitting;
using Accord.Statistics.Models.Regression.Linear;
using System;
using System.Collections.Generic;
using System.Linq;
using CsvHelper;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace Accord_Test
{
    class Program
    {
        public static readonly string TRAINING_DATA_FILE = @"..\..\training_data_3.xls";
        public static readonly string TEST_DATA_FILE = @"..\..\training_data_2.xls";

        static void Main(string[] args)
        {
            var metsRecords = LoadDataRecords(@"..\..\training_data_3.xls", 0);

            var expr3 = Experiment3(90, 0.92338482m, 0.91712924m, 100, 0.93793195m, 0.91501673m);
            Experiment4(expr3);

            SaveToCsv();

            Console.In.ReadLine();
        }

        private static void SaveToCsv()
        {
            var metsRecords = LoadDataRecords(TRAINING_DATA_FILE, 0);
            using (var tw = File.CreateText($"d:\\training_data.csv"))
            {
                var csv = new CsvHelper.CsvWriter(tw);
                csv.WriteHeader<MetsDataRecord>();
                csv.NextRecord();
                csv.WriteRecords(metsRecords);
            }
        }

        private static void Experiment2(int neuronCountPPV, int neuronCountNPV, int repeats)
        {
            var metsRecords = LoadDataRecords(TRAINING_DATA_FILE, 0);

            var results = new List<ExperimentPredictiveValues>();

            DateTime et = DateTime.Now;

            for (int i = 0; i < repeats; i++)
            {
                var es = DateTime.Now;

                var data = CreateData(metsRecords, 0.8, 0.1);

                Console.Out.WriteLine("================================");
                Console.Out.WriteLine($"       EXPERIMENT #{i:000}");
                Console.Out.WriteLine("================================");
                Console.Out.WriteLine();

                Console.Out.WriteLine("================================");
                Console.Out.WriteLine($"    ANN PPV, NC:{neuronCountPPV}");
                Console.Out.WriteLine("================================");

                var ann_PPV_PV = Ann(data, neuronCountPPV, null, out var ann);
                LogAPV(ann_PPV_PV);

                var ann_NPV_PV = new AlgoritamPredictiveValues();
                if (neuronCountNPV != neuronCountPPV)
                {
                    Console.Out.WriteLine("================================");
                    Console.Out.WriteLine($"     ANN NPV, NC:{neuronCountNPV}");
                    Console.Out.WriteLine("================================");

                    ann_NPV_PV = Ann(data, neuronCountNPV, null, out ann);
                    LogAPV(ann_NPV_PV);
                }

                var experimentPV = new ExperimentPredictiveValues()
                {
                    AnnPpvLearnNPV = ann_PPV_PV.LearnNPV,
                    AnnPpvLearnPPV = ann_PPV_PV.LearnPPV,
                    AnnPpvLearnSENS = ann_PPV_PV.LearnSENS,
                    AnnPpvLearnSPEC = ann_PPV_PV.LearnSPEC,
                    AnnPpvValidateNPV = ann_PPV_PV.ValidateNPV,
                    AnnPpvValidatePPV = ann_PPV_PV.ValidatePPV,
                    AnnPpvValidateSENS = ann_PPV_PV.ValidateSENS,
                    AnnPpvValidateSPEC = ann_PPV_PV.ValidateSPEC,
                    AnnPpvTestNPV = ann_PPV_PV.TestNPV,
                    AnnPpvTestPPV = ann_PPV_PV.TestPPV,
                    AnnPpvTestSENS = ann_PPV_PV.TestSENS,
                    AnnPpvTestSPEC = ann_PPV_PV.TestSPEC,

                    AnnNpvLearnNPV = ann_NPV_PV.LearnNPV,
                    AnnNpvLearnPPV = ann_NPV_PV.LearnPPV,
                    AnnNpvLearnSENS = ann_NPV_PV.LearnSENS,
                    AnnNpvLearnSPEC = ann_NPV_PV.LearnSPEC,
                    AnnNpvValidateNPV = ann_NPV_PV.ValidateNPV,
                    AnnNpvValidatePPV = ann_NPV_PV.ValidatePPV,
                    AnnNpvValidateSENS = ann_NPV_PV.ValidateSENS,
                    AnnNpvValidateSPEC = ann_NPV_PV.ValidateSPEC,
                    AnnNpvTestNPV = ann_NPV_PV.TestNPV,
                    AnnNpvTestPPV = ann_NPV_PV.TestPPV,
                    AnnNpvTestSENS = ann_NPV_PV.TestSENS,
                    AnnNpvTestSPEC = ann_NPV_PV.TestSPEC,
                };

                LogEPV(experimentPV);

                results.Add(experimentPV);

                LogMessage($"Experiment time: {DateTime.Now - es}, total time {DateTime.Now - et}");
            }

            LogAvgEPV(results);

            using (var tw = File.CreateText($"d:\\mets_experiment_2_{neuronCountPPV}_{neuronCountNPV}_R{repeats}{DateTime.Now:yyyy_MM_dd_hh_mm}.csv"))
            {
                var csv = new CsvHelper.CsvWriter(tw);
                csv.WriteHeader<ExperimentPredictiveValues>();
                csv.NextRecord();
                csv.WriteRecords(results);
            }
        }

        private static Experiment3Output Experiment3(int neuronCount, decimal minAnnPPV, decimal minAnnNPV, int treeCount, decimal minForestPPV, decimal minForestNPV)
        {
            var result = new Experiment3Output();

            result.NeuronCount = neuronCount;

            var metsRecords = LoadDataRecords(TRAINING_DATA_FILE, 0);

            DateTime et = DateTime.Now;

            var currentMinPPV = 0m;
            var currentMinNPV = 0m;

            Data data = null;

            var learningData = new List<AnnLearningRecord>();
            var annPV = new AlgoritamPredictiveValues();
            var forestPV = new AlgoritamPredictiveValues();

            while (currentMinPPV < minAnnPPV || currentMinNPV < minAnnNPV || currentMinPPV <= currentMinNPV || currentMinPPV < minForestPPV || currentMinNPV < minForestNPV)
            {
                var es = DateTime.Now;

                data = CreateData(metsRecords, 0.8, 0.1);

                result.Means = data.Means;
                result.StandardDeviations = data.StandardDeviations;

                Console.Out.WriteLine("================================");
                Console.Out.WriteLine($"    ANN PPV, NC:{neuronCount}");
                Console.Out.WriteLine("================================");

                learningData.Clear();
                annPV = Ann(data, neuronCount, learningData, out var ann);
                currentMinPPV = annPV.TestPPV;
                currentMinNPV = annPV.TestNPV;

                result.Ann = ann;

                LogAPV(annPV);

                LogMessage($"Experiment time: {DateTime.Now - es}, total time {DateTime.Now - et}");
            }

                var experimentPV = new ExperimentPredictiveValues()
                {
                    AnnPpvLearnNPV = annPV.LearnNPV,
                    AnnPpvLearnPPV = annPV.LearnPPV,
                    AnnPpvValidateNPV = annPV.ValidateNPV,
                    AnnPpvValidatePPV = annPV.ValidatePPV,
                    AnnPpvTestNPV = annPV.TestNPV,
                    AnnPpvTestPPV = annPV.TestPPV,
                };


            using (var tw = File.CreateText($"d:\\mets_experiment_3_{neuronCount}_{DateTime.Now:yyyy_MM_dd_hh_mm}_rndfrst.csv"))
            {
                var csv = new CsvHelper.CsvWriter(tw);
                csv.WriteHeader<AnnLearningRecord>();
                csv.NextRecord();
                csv.WriteRecords(learningData);
            }

            using (var tw = File.CreateText($"d:\\mets_experiment_3_PV_{neuronCount}_{DateTime.Now:yyyy_MM_dd_hh_mm}_rndfrst.csv"))
            {
                var csv = new CsvHelper.CsvWriter(tw);
                csv.WriteHeader<ExperimentPredictiveValues>();
                csv.NextRecord();
                csv.WriteRecords(new ExperimentPredictiveValues[] { experimentPV });
            }

            return result;
        }

        private static void Experiment4(Experiment3Output exp3)
            {
            var metsRecords = LoadDataRecords(TEST_DATA_FILE, 0);

            for (int i = 0; i < metsRecords.Count; i++)
            {
                var mets = CalculateMets(metsRecords[i]);
                if (metsRecords[i].METS != mets)
                    throw new Exception("Invalid data");
            }

            if (metsRecords.Count > 100)
            {
                var mr = new List<MetsDataRecord>();

                var rnd = new Random();
                while (mr.Count < 100)
                {
                    var index = rnd.Next(metsRecords.Count);
                    mr.Add(metsRecords[index]);
                    metsRecords.RemoveAt(index);
                }
                metsRecords = mr;
            }

            var inputs = CreateInputs(metsRecords).ToArray();
            inputs = inputs.ZScores(exp3.Means, exp3.StandardDeviations);
            var outputs = CreateOutputs(metsRecords).ToArray();

            var annPredictions = new double[inputs.Length][];

            for (int i = 0; i < inputs.Length; i++)
            {
                var annr = exp3.Ann.Compute(inputs[i])[0];
                var to = Math.Round(outputs[i][0], 0);
                var anno = Math.Round(annr, 0);

                annPredictions[i] = new double[] { anno };
            }



            var annPV = CalculatePredictiveValues(outputs, annPredictions);

            var results = new List<Experiment4ResultRecord>();
            for (int i = 0; i < metsRecords.Count; i++)
            {
                results.Add(new Experiment4ResultRecord()
                {
                    AGE = metsRecords[i].AGE,
                    BMI = metsRecords[i].BMI,
                    DBP = metsRecords[i].DBP,
                    FPG = metsRecords[i].FPG,
                    GEN = metsRecords[i].GEN,
                    HDL = metsRecords[i].HDL,
                    HT = metsRecords[i].HT,
                    METS = metsRecords[i].METS,
                    METS_PR_ANN = (int)Math.Round(annPredictions[i][0]),
                    SBP = metsRecords[i].SBP,
                    TG = metsRecords[i].TG,
                    WC = metsRecords[i].WC,
                    WHtR = metsRecords[i].WHtR,
                    WT = metsRecords[i].WT
                });
            }

            var experimentPV = new Experiment4PredictiveValues()
            {
                AnnNPV = annPV.NPV,
                AnnPPV = annPV.PPV,
            };


            using (var tw = File.CreateText($"d:\\mets_experiment_4_{exp3.NeuronCount}_{DateTime.Now:yyyy_MM_dd_hh_mm}.csv"))
            {
                var csv = new CsvHelper.CsvWriter(tw);
                csv.WriteHeader<Experiment4ResultRecord>();
                csv.NextRecord();
                csv.WriteRecords(results);
            }

            using (var tw = File.CreateText($"d:\\mets_experiment_4_PV_{exp3.NeuronCount}_{DateTime.Now:yyyy_MM_dd_hh_mm}.csv"))
            {
                var csv = new CsvHelper.CsvWriter(tw);
                csv.WriteHeader<Experiment4PredictiveValues>();
                csv.NextRecord();
                csv.WriteRecords(new Experiment4PredictiveValues[] { experimentPV });
        }
        }

        private const int MALE = 1;
        private const int FEMALE = 2;
        private static int CalculateMets(MetsDataRecord mr)
        {
            var mets = 0;
            if ((mr.BMI > 30) || ((mr.GEN == FEMALE && mr.WC >= 80) || (mr.GEN == MALE && mr.WC >= 94)))
            {
                var p = 0;
                if (mr.SBP >= 130 || mr.DBP >= 85)
                    p += 1;
                if ((mr.GEN == FEMALE && mr.HDL < 1.29m) || (mr.GEN == MALE && mr.HDL < 1.03m))
                    p += 1;
                if (mr.TG >= 1.7m)
                    p += 1;
                if (mr.FPG >= 5.6m)
                    p += 1;
                if (p >= 2)
                    mets = 1;
            }
            return mets;
        }

        private static void Experiment1()
        {
            Console.WriteLine("Start neuron count:");
            var startNeuronCount = int.Parse(Console.ReadLine());
            Console.WriteLine("End neuron count:");
            var endNeuronCount = int.Parse(Console.ReadLine());
            Console.WriteLine("Repeats:");
            var repeats = int.Parse(Console.ReadLine());

            var metsRecords = LoadDataRecords(TRAINING_DATA_FILE, 0);

            var results = new List<Experiment2Results>();

            DateTime et = DateTime.Now;

            for (int neuronCount = startNeuronCount; neuronCount <= endNeuronCount; neuronCount++)
            {
                Console.Out.WriteLine("================================");
                Console.Out.WriteLine($"       NEURON COUNT #{neuronCount:000}");
                Console.Out.WriteLine("================================");

                var iterationResults = new List<AlgoritamPredictiveValues>();

                for (int i = 0; i < repeats; i++)
                {
                    var es = DateTime.Now;

                    var data = CreateData(metsRecords, 0.8, 0.1);

                    Console.Out.WriteLine("================================");
                    Console.Out.WriteLine($"       EXPERIMENT #{i + 1:000}, NC{neuronCount:000}");
                    Console.Out.WriteLine("================================");
                    Console.Out.WriteLine();

                    var annPV = Ann(data, neuronCount, null, out var ann);
                    LogAPV(annPV);

                    iterationResults.Add(annPV);

                    LogMessage($"Experiment time: {DateTime.Now - es}, total time {DateTime.Now - et}");
                }

                results.Add(new Experiment2Results()
                {
                    NeuronCount = neuronCount,
                    LearnNPV = iterationResults.Average(x => x.LearnNPV),
                    LearnPPV = iterationResults.Average(x => x.LearnPPV),
                    ValidateNPV = iterationResults.Average(x => x.ValidateNPV),
                    ValidatePPV = iterationResults.Average(x => x.ValidatePPV),
                    TestNPV = iterationResults.Average(x => x.TestNPV),
                    TestPPV = iterationResults.Average(x => x.TestPPV),
                });
            }

            using (var tw = File.CreateText($"d:\\mets_experiment_2_{startNeuronCount}_{endNeuronCount}_R{repeats}_{DateTime.Now:yyyy_MM_dd_hh_mm}.csv"))
            {
                var csv = new CsvHelper.CsvWriter(tw);
                csv.WriteHeader<Experiment2Results>();
                csv.NextRecord();
                csv.WriteRecords(results);
            }
        }


        private static void LogAPV(AlgoritamPredictiveValues annPV)
        {
            Console.Out.WriteLine("================================");

            Console.Out.WriteLine($"LEARN PPV:     {annPV.LearnPPV}");
            Console.Out.WriteLine($"LEARN NPV:     {annPV.LearnNPV}");
            Console.Out.WriteLine($"LEARN SPEC:    {annPV.LearnSPEC}");
            Console.Out.WriteLine($"LEARN SENS:    {annPV.LearnSENS}");
            Console.Out.WriteLine($"VALIDATE PPV:  {annPV.ValidatePPV}");
            Console.Out.WriteLine($"VALIDATE NPV:  {annPV.ValidateNPV}");
            Console.Out.WriteLine($"VALIDATE SPEC: {annPV.ValidateSPEC}");
            Console.Out.WriteLine($"VALIDATE SENS: {annPV.ValidateSENS}");
            Console.Out.WriteLine($"TEST PPV:      {annPV.TestPPV}");
            Console.Out.WriteLine($"TEST NPV:      {annPV.TestNPV}");
            Console.Out.WriteLine($"TEST SPEC:     {annPV.TestSPEC}");
            Console.Out.WriteLine($"TEST SENS:     {annPV.TestSENS}");

            Console.Out.WriteLine("================================");
        }

        private static void LogEPV(ExperimentPredictiveValues ePV)
        {
            Console.Out.WriteLine("================================");

            Console.Out.WriteLine($"ANN PPV LEARN PPV:    {ePV.AnnPpvLearnPPV}");
            Console.Out.WriteLine($"ANN PPV LEARN NPV:    {ePV.AnnPpvLearnNPV}");
            Console.Out.WriteLine($"ANN PPV VALIDATE PPV: {ePV.AnnPpvValidatePPV}");
            Console.Out.WriteLine($"ANN PPV VALIDATE NPV: {ePV.AnnPpvValidateNPV}");
            Console.Out.WriteLine($"ANN PPV TEST PPV:     {ePV.AnnPpvTestPPV}");
            Console.Out.WriteLine($"ANN PPV TEST NPV:     {ePV.AnnPpvTestNPV}");

            Console.Out.WriteLine($"ANN NPV LEARN PPV:    {ePV.AnnNpvLearnPPV}");
            Console.Out.WriteLine($"ANN NPV LEARN NPV:    {ePV.AnnNpvLearnNPV}");
            Console.Out.WriteLine($"ANN NPV VALIDATE PPV: {ePV.AnnNpvValidatePPV}");
            Console.Out.WriteLine($"ANN NPV VALIDATE NPV: {ePV.AnnNpvValidateNPV}");
            Console.Out.WriteLine($"ANN NPV TEST PPV:     {ePV.AnnNpvTestPPV}");
            Console.Out.WriteLine($"ANN NPV TEST NPV:     {ePV.AnnNpvTestNPV}");

            Console.Out.WriteLine("================================");
        }

        private static void LogAvgEPV(List<ExperimentPredictiveValues> ePV)
        {
            Console.Out.WriteLine("================================");

            Console.Out.WriteLine($"ANN PPV LEARN PPV:    {ePV.Average(x => x.AnnPpvLearnPPV)}");
            Console.Out.WriteLine($"ANN PPV LEARN NPV:    {ePV.Average(x => x.AnnPpvLearnNPV)}");
            Console.Out.WriteLine($"ANN PPV VALIDATE PPV: {ePV.Average(x => x.AnnPpvValidatePPV)}");
            Console.Out.WriteLine($"ANN PPV VALIDATE NPV: {ePV.Average(x => x.AnnPpvValidateNPV)}");
            Console.Out.WriteLine($"ANN PPV TEST PPV:     {ePV.Average(x => x.AnnPpvTestPPV)}");
            Console.Out.WriteLine($"ANN PPV TEST NPV:     {ePV.Average(x => x.AnnPpvTestNPV)}");

            Console.Out.WriteLine($"ANN NPV LEARN PPV:    {ePV.Average(x => x.AnnNpvLearnPPV)}");
            Console.Out.WriteLine($"ANN NPV LEARN NPV:    {ePV.Average(x => x.AnnNpvLearnNPV)}");
            Console.Out.WriteLine($"ANN NPV VALIDATE PPV: {ePV.Average(x => x.AnnNpvValidatePPV)}");
            Console.Out.WriteLine($"ANN NPV VALIDATE NPV: {ePV.Average(x => x.AnnNpvValidateNPV)}");
            Console.Out.WriteLine($"ANN NPV TEST PPV:     {ePV.Average(x => x.AnnNpvTestPPV)}");
            Console.Out.WriteLine($"ANN NPV TEST NPV:     {ePV.Average(x => x.AnnNpvTestNPV)}");

            Console.Out.WriteLine("================================");
        }

        private static void Statistics(Data data, double[] results, string caption)
        {
            var truePositive = 0;
            var falsePositive = 0;
            var trueNegative = 0;
            var falseNegative = 0;

            for (int i = 0; i < data.TestInputs.Length; i++)
            {
                var to = Math.Round(data.TestOutputs[i][0], 0);
                var anno = results[i];
                if (anno == to)
                {
                    if (to == 0)
                        trueNegative++;
                    else
                        truePositive++;
                }
                else
                {
                    if (to == 0)
                        falseNegative++;
                    else
                        falsePositive++;
                }
            }

            Console.Out.WriteLine("================================");
            Console.Out.WriteLine($" {caption}");
            Console.Out.WriteLine("================================");

            Console.Out.WriteLine($"TRUE POSITIVE COUNT: {truePositive}");
            Console.Out.WriteLine($"TRUE NEGATIVE COUNT: {trueNegative}");

            Console.Out.WriteLine($"FALSE POSITIVE COUNT: {falsePositive}");
            Console.Out.WriteLine($"FALSE NEGATIVE COUNT: {falseNegative}");

            Console.Out.WriteLine($"PPV: {(double)truePositive / ((double)truePositive + (double)falsePositive)}");
            Console.Out.WriteLine($"NPV: {(double)trueNegative / ((double)trueNegative + (double)falseNegative)}");

            Console.Out.WriteLine("================================");
        }

        private const int INPUT_COUNT = 6;
        private const int OUTPUT_COUNT = 1;

        private static AlgoritamPredictiveValues Ann(Data data, int neuronCount, List<AnnLearningRecord> learningData, out ActivationNetwork resultAnn)
        {
            // za aktivacionu funkciju uzimamo standardnu sigmoidnu
            var activationFunction = new SigmoidFunction();

            // kreiramo neuronsku mrežu sa 6 neurona u ulazniom sloju i jednim u izlaznom
            // broj skrivenih neurona je određen parametrom neuronCount
            var ann = new ActivationNetwork(activationFunction, INPUT_COUNT, neuronCount, OUTPUT_COUNT);

            // postavljanje inicijalnih težinski faktora
            new NguyenWidrow(ann).Randomize();

            // algoritam za obučavanje
            var teacher = new LevenbergMarquardtLearning(ann);

            teacher.LearningRate = 0.001;

            // uključujemo regularizaciju
            teacher.UseRegularization = true;

            double err;
            int count = 0;
            int failCount = 0;
            PredictiveValues lastPredictiveValues = null;
            int epochCounter = 1;
            while (true)
            {
                err = teacher.RunEpoch(data.LearnInputs, data.LearnOutputs);

                // izracunavamo prediktivne vrednosti
                var pv = CalculatePredictiveValues(ann, data);

                LogMessage($"Epoch #{count + 1:000}, PPV: {pv.PPV,10:#0.000000}, NPV: {pv.NPV,10:#0.000000}, SPEC: {pv.SPEC,10:#0.000000}, SENS: {pv.SENS,10:#0.000000}");

                if (learningData != null)
                {
                    learningData.Add(new AnnLearningRecord()
                    {
                        EpochNumber = epochCounter++,
                        NPV = pv.NPV,
                        PPV = pv.PPV,
                        N_FN = pv.N_FN,
                        N_FP = pv.N_FP,
                        N_TN = pv.N_TN,
                        N_TP = pv.N_TP
                    });
                }

                if (lastPredictiveValues != null)
                {
                    // proveravamo da li se performansa poboljsala ili ne
                    if (lastPredictiveValues.PPV >= pv.PPV && lastPredictiveValues.NPV >= pv.NPV)
                    {
                        failCount++;
                        // obuka se prekida ukoliko do poboljsanja nije doslo u poslednjih 10 epoha
                        if (failCount >= 10)
                            break;
                    }
                    else
                        failCount = 0;
                }

                lastPredictiveValues = pv;

                count++;
                if (count >= 500)
                    break;

            }

            // izracunavanje izlaza
            double[][] AnnCompute(double[][] inputs, double[][] outputs)
            {
                var res = new double[inputs.Length][];

                for (int i = 0; i < inputs.Length; i++)
                {
                    var annr = ann.Compute(inputs[i])[0];
                    var to = Math.Round(outputs[i][0], 0);
                    var anno = Math.Round(annr, 0);

                    res[i] = new double[] { anno };
                }

                return res;
            }

            var learnOutputs = AnnCompute(data.LearnInputs, data.LearnOutputs);
            var validateOutputs = AnnCompute(data.ValidateInputs, data.ValidateOutputs);
            var testOutputs = AnnCompute(data.TestInputs, data.TestOutputs);

            var learnPV = CalculatePredictiveValues(data.LearnOutputs, learnOutputs);
            var validatePV = CalculatePredictiveValues(data.ValidateOutputs, validateOutputs);
            var testPV = CalculatePredictiveValues(data.TestOutputs, testOutputs);

            var result = new AlgoritamPredictiveValues()
            {
                LearnNPV = learnPV.NPV,
                LearnPPV = learnPV.PPV,
                LearnSENS = learnPV.SENS,
                LearnSPEC = learnPV.SPEC,

                ValidateNPV = validatePV.NPV,
                ValidatePPV = validatePV.PPV,
                ValidateSENS = validatePV.SENS,
                ValidateSPEC = validatePV.SPEC,

                TestNPV = testPV.NPV,
                TestPPV = testPV.PPV,
                TestSENS = testPV.SENS,
                TestSPEC = testPV.SPEC,
            };

            resultAnn = ann;

            return result;
        }

        private static void LogMessage(string msg)
        {
            Console.WriteLine(msg);
        }

        private static decimal CalcAnnPerformance(ActivationNetwork ann, Data data)
        {
            var annOutputs = new double[data.ValidateOutputs.Length][];
            for (int i = 0; i < data.ValidateInputs.Length; i++)
            {
                var vi = data.ValidateInputs[i];
                annOutputs[i] = new double[] { Math.Round(ann.Compute(vi)[0], 0) };
            }

            CalculatePredictiveValues(data.ValidateOutputs, annOutputs);

            return (decimal)Math.Round(new SquareLoss(data.ValidateOutputs).Loss(annOutputs), 6);
        }

        private static PredictiveValues CalculatePredictiveValues(ActivationNetwork ann, Data data)
        {
            var annOutputs = new double[data.ValidateOutputs.Length][];
            for (int i = 0; i < data.ValidateInputs.Length; i++)
            {
                var vi = data.ValidateInputs[i];
                annOutputs[i] = new double[] { Math.Round(ann.Compute(vi)[0], 0) };
            }

            return CalculatePredictiveValues(data.ValidateOutputs, annOutputs);
        }

        private static PredictiveValues CalculatePredictiveValues(double[][] expectedOutputs, double[][] actualOutputs)
        {
            var n_tp = 0;
            var n_fp = 0;
            var n_tn = 0;
            var n_fn = 0;
            var n = expectedOutputs.Length;

            for (int i = 0; i < n; i++)
            {
                var expected = (int)Math.Round(expectedOutputs[i][0]);
                var actual = (int)Math.Round(actualOutputs[i][0]);
                if (actual == expected)
                {
                    // u ovom slucaju imamo tacnu predikciju
                    // potrebno je odrediti da li pozitivna ili negativna
                    if (expected == 0)
                        n_tn++;  // negativna
                    else
                        n_tp++;  // pozitivna
                }
                else
                {
                    // u ovom slucaju imamo netacnu predikciju
                    // potrebno je odrediti da li pozitivna ili negativna
                    if (expected == 0)
                        n_fn++;  // negativna
                    else
                        n_fp++;  // pozitivna
                }
            }

            var result = new PredictiveValues()
            {
                N_TP = n_tp,
                N_TN = n_tn,
                N_FP = n_fp,
                N_FN = n_fn,
                SENS = (decimal)(n_tp / (double)(n_tp + n_fn)),
                SPEC = (decimal)(n_tn / (double)(n_tn + n_fp)),
                PPV = (decimal)Math.Round(n_tp / (n_tp + (double)n_fp), 6),
                NPV = (decimal)Math.Round(n_tn / (n_tn + (double)n_fn), 6)
            };

            return result;
        }

        //var varsList = new List<DecisionVariable>();
        //varsList.Add(DecisionVariable.Discrete("GEN", new Accord.IntRange(1, 2)));
        //varsList.Add(DecisionVariable.Discrete(
        //    "AGE", new Accord.IntRange(
        //        data.LearnInputs.Min(x => (int)Math.Round(x[1])),
        //        data.LearnInputs.Max(x => (int)Math.Round(x[1])))));
        //varsList.Add(DecisionVariable.Continuous(
        //    "BMI", new Accord.DoubleRange(
        //        data.LearnInputs.Min(x => x[2]),
        //        data.LearnInputs.Max(x => x[2]))));
        //varsList.Add(DecisionVariable.Continuous(
        //    "WHtR", new Accord.DoubleRange(
        //        data.LearnInputs.Min(x => x[3]),
        //        data.LearnInputs.Max(x => x[3]))));

        //varsList.Add(DecisionVariable.Discrete(
        //    "SBP", new Accord.IntRange(
        //        data.LearnInputs.Min(x => (int)Math.Round(x[4])),
        //        data.LearnInputs.Max(x => (int)Math.Round(x[4])))));

        //varsList.Add(DecisionVariable.Discrete(
        //    "DBP", new Accord.IntRange(
        //        data.LearnInputs.Min(x => (int)Math.Round(x[5])),
        //        data.LearnInputs.Max(x => (int)Math.Round(x[5])))));


        private static List<MetsDataRecord> LoadDataRecords(string fileName, int sheetNumber)
        {
            var reader = new ExcelReader(fileName, true);
            var data = reader.GetWorksheet(sheetNumber);

            var cols = reader.GetColumnsList(reader.GetWorksheetList()[sheetNumber]);
            var fpg = cols.Contains("FPG");
            var hdl = cols.Contains("HDL");
            var tg = cols.Contains("TG");
            var wc = cols.Contains("WC");
            var ht = cols.Contains("HT");
            var wt = cols.Contains("WT");
            var mets = cols.Contains("METS");
            var result = new List<MetsDataRecord>();
            for (int i = 0; i < data.Rows.Count; i++)
            {
                result.Add(new MetsDataRecord()
                {
                    GEN = Convert.ToInt32(data.Rows[i]["GEN"]),
                    AGE = Convert.ToInt32(data.Rows[i]["AGE"]),
                    BMI = Convert.ToDecimal(data.Rows[i]["BMI"]),
                    WHtR = Convert.ToDecimal(data.Rows[i]["WHtR"]),
                    SBP = Convert.ToInt32(data.Rows[i]["SBP"]),
                    DBP = Convert.ToInt32(data.Rows[i]["DBP"]),
                    METS = mets ? Convert.ToInt32(data.Rows[i]["METS"]) : 0,
                    FPG = fpg ? Convert.ToDecimal(data.Rows[i]["FPG"]) : 0,
                    HDL = hdl ? Convert.ToDecimal(data.Rows[i]["HDL"]) : 0,
                    TG = tg ? Convert.ToDecimal(data.Rows[i]["TG"]) : 0,
                    WC = wc ? Convert.ToDecimal(data.Rows[i]["WC"]) : 0,
                    HT = ht ? Convert.ToDecimal(data.Rows[i]["HT"]) : 0,
                    WT = wt ? Convert.ToDecimal(data.Rows[i]["WT"]) : 0,
                });
            }

            if (!mets && tg && hdl && fpg && wc)
            {
                foreach (var r in result)
                    r.METS = CalculateMets(r);
            }
            return result;
        }

        public static Data CreateData(List<MetsDataRecord> metsData, double learnPercent, double validatePercent)
        {
            var result = new Data();

            var inputs = CreateInputs(metsData);
            var outputs = CreateOutputs(metsData);

            result.Means = Measures.Mean(inputs.ToArray(), 0);
            result.StandardDeviations = Measures.StandardDeviation(inputs.ToArray(), result.Means);

            //inputs.ToArray().Center(true);
            //inputs.ToArray().Standardize(true);

            inputs = new List<double[]>(inputs.ToArray().ZScores());

            int learnCount = (int)(learnPercent * metsData.Count);
            int validateCount = (int)(validatePercent * metsData.Count);
            int testCount = metsData.Count - learnCount - validateCount;

            var rnd = new Random();

            var learnInputs = new List<double[]>();
            var learnOutputs = new List<double[]>();

            for (int i = 0; i < learnCount; i++)
            {
                var index = rnd.Next(inputs.Count);
                learnInputs.Add(inputs[index]);
                learnOutputs.Add(outputs[index]);
                inputs.RemoveAt(index);
                outputs.RemoveAt(index);
            }

            var validateInputs = new List<double[]>();
            var validateOutputs = new List<double[]>();

            for (int i = 0; i < validateCount; i++)
            {
                var index = rnd.Next(inputs.Count);
                validateInputs.Add(inputs[index]);
                validateOutputs.Add(outputs[index]);
                inputs.RemoveAt(index);
                outputs.RemoveAt(index);
            }


            var testInputs = new List<double[]>();
            var testOutputs = new List<double[]>();

            for (int i = 0; i < testCount; i++)
            {
                var index = rnd.Next(inputs.Count);
                testInputs.Add(inputs[index]);
                testOutputs.Add(outputs[index]);
                inputs.RemoveAt(index);
                outputs.RemoveAt(index);
            }

            result.LearnInputs = learnInputs.ToArray();
            result.LearnOutputs = learnOutputs.ToArray();

            result.ValidateInputs = validateInputs.ToArray();
            result.ValidateOutputs = validateOutputs.ToArray();

            result.TestInputs = testInputs.ToArray();
            result.TestOutputs = testOutputs.ToArray();

            return result;
        }

        private static List<double[]> CreateInputs(List<MetsDataRecord> data)
        {
            var result = new List<double[]>();

            for (int i = 0; i < data.Count; i++)
            {
                var row = data[i];
                result.Add(new double[] { row.GEN, row.AGE, (double)row.BMI, (double)row.WHtR, row.SBP, row.DBP });
            }

            return result;
        }

        private static List<double[]> CreateOutputs(List<MetsDataRecord> data)
        {
            var result = new List<double[]>();

            for (int i = 0; i < data.Count; i++)
            {
                var row = data[i];
                result.Add(new double[] { row.METS });
            }

            return result;
        }
    }

    public class AnnLearningRecord
    {
        public int EpochNumber { get; set; }
        public int N_TP { get; set; }
        public int N_TN { get; set; }
        public int N_FP { get; set; }
        public int N_FN { get; set; }
        public decimal PPV { get; set; }
        public decimal NPV { get; set; }
    }

    public class Data
    {
        public double[] Means;
        public double[] StandardDeviations;

        public double[][] LearnInputs;
        public double[][] LearnOutputs;

        public double[][] ValidateInputs;
        public double[][] ValidateOutputs;

        public double[][] TestInputs;
        public double[][] TestOutputs;
    }

    public class MetsDataRecord
    {
        public int GEN { get; set; }
        public int AGE { get; set; }
        public decimal BMI { get; set; }
        public decimal WHtR { get; set; }
        public int SBP { get; set; }
        public int DBP { get; set; }
        public int METS { get; set; }

        public decimal HT { get; set; }
        public decimal WT { get; set; }
        public decimal WC { get; set; }
        public decimal TG { get; set; }
        public decimal HDL { get; set; }
        public decimal FPG { get; set; }
    }

    public class MetsDataRecordComparer : IEqualityComparer<MetsDataRecord>
    {
        public bool Equals(MetsDataRecord x, MetsDataRecord y)
        {
            return x.AGE == y.AGE
                && x.BMI == y.BMI
                && x.DBP == y.DBP
                && x.SBP == y.SBP
                && x.WHtR == y.WHtR
                && x.GEN == y.GEN;
        }

        public int GetHashCode(MetsDataRecord obj)
        {
            return obj.GEN.GetHashCode()
                ^ obj.BMI.GetHashCode()
                ^ obj.AGE.GetHashCode()
                ^ obj.DBP.GetHashCode()
                ^ obj.SBP.GetHashCode()
                ^ obj.WHtR.GetHashCode();
        }
    }

    public class PredictiveValues
    {
        public int N_TP { get; set; }
        public int N_TN { get; set; }
        public int N_FP { get; set; }
        public int N_FN { get; set; }
        public decimal PPV { get; set; }
        public decimal NPV { get; set; }
        public decimal SENS { get; set; }
        public decimal SPEC { get; set; }
    }


public class AlgoritamPredictiveValues
    {
        public decimal LearnPPV;
        public decimal LearnNPV;
        public decimal LearnSENS;
        public decimal LearnSPEC;

        public decimal ValidatePPV;
        public decimal ValidateNPV;
        public decimal ValidateSENS;
        public decimal ValidateSPEC;

        public decimal TestPPV;
        public decimal TestNPV;
        public decimal TestSENS;
        public decimal TestSPEC;
    }

    public class Experiment2Results
    {
        public int NeuronCount { get; set; }
        public decimal LearnPPV { get; set; }
        public decimal LearnNPV { get; set; }
        public decimal ValidatePPV { get; set; }
        public decimal ValidateNPV { get; set; }
        public decimal TestPPV { get; set; }
        public decimal TestNPV { get; set; }

    }

    public class Experiment5Results
    {
        public int TreeCount { get; set; }
        public decimal LearnPPV { get; set; }
        public decimal LearnNPV { get; set; }
        public decimal ValidatePPV { get; set; }
        public decimal ValidateNPV { get; set; }
        public decimal TestPPV { get; set; }
        public decimal TestNPV { get; set; }

    }

    public class ExperimentPredictiveValues
    {
        public decimal AnnPpvLearnPPV { get; set; }
        public decimal AnnPpvLearnNPV { get; set; }
        public decimal AnnPpvLearnSPEC { get; set; }
        public decimal AnnPpvLearnSENS { get; set; }
        public decimal AnnPpvValidatePPV { get; set; }
        public decimal AnnPpvValidateNPV { get; set; }
        public decimal AnnPpvValidateSPEC { get; set; }
        public decimal AnnPpvValidateSENS { get; set; }
        public decimal AnnPpvTestPPV { get; set; }
        public decimal AnnPpvTestNPV { get; set; }
        public decimal AnnPpvTestSPEC { get; set; }
        public decimal AnnPpvTestSENS { get; set; }


        public decimal AnnNpvLearnPPV { get; set; }
        public decimal AnnNpvLearnNPV { get; set; }
        public decimal AnnNpvLearnSPEC { get; set; }
        public decimal AnnNpvLearnSENS { get; set; }
        public decimal AnnNpvValidatePPV { get; set; }
        public decimal AnnNpvValidateNPV { get; set; }
        public decimal AnnNpvValidateSPEC { get; set; }
        public decimal AnnNpvValidateSENS { get; set; }
        public decimal AnnNpvTestPPV { get; set; }
        public decimal AnnNpvTestNPV { get; set; }
        public decimal AnnNpvTestSPEC { get; set; }
        public decimal AnnNpvTestSENS { get; set; }
    }

    public class Experiment3Output
    {
        public ActivationNetwork Ann { get; set; }
        public double[] Means { get; set; }
        public double[] StandardDeviations { get; set; }
        public int NeuronCount { get; set; }
    }

    public class Experiment4ResultRecord
    {
        public int GEN { get; set; }
        public int AGE { get; set; }
        public decimal BMI { get; set; }
        public decimal WT { get; set; }
        public decimal HT { get; set; }
        public decimal WC { get; set; }
        public decimal WHtR { get; set; }
        public int SBP { get; set; }
        public int DBP { get; set; }
        public decimal TG { get; set; }
        public decimal HDL { get; set; }
        public decimal FPG { get; set; }
        public int METS { get; set; }
        public int METS_PR_ANN { get; set; }
    }


    public class Experiment4PredictiveValues
    {
        public decimal AnnPPV { get; set; }
        public decimal AnnNPV { get; set; }
    }
}
