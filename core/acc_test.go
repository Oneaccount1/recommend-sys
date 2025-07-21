package core

import (
	"gonum.org/v1/gonum/stat"
	"math"
	"testing"
)

const EPSILON float64 = 0.05

func Evaluate(t *testing.T, algo Algorithm, dataSet TrainSet,
	expectRMSE float64, expectMAE float64) {
	results := CrossValidate(algo, dataSet, []Metrics{RootMeanSquareError, MeanAbsoluteError}, 5, 0, nil)
	// Cross validation
	// Check RMSE
	rmse := stat.Mean(results[0], nil)
	if math.Abs(rmse-expectRMSE) > EPSILON {
		t.Fatalf("RMSE(%.3f) not in %.3f±%.3f", rmse, expectRMSE, EPSILON)
	}
	// Check MAE
	mae := stat.Mean(results[1], nil)
	if math.Abs(mae-expectMAE) > EPSILON {
		t.Fatalf("MAE(%.3f) not in %.3f±%.3f", mae, expectMAE, EPSILON)
	}
}

func TestRandom(t *testing.T) {
	Evaluate(t, NewRandom(), LoadDataFromBuiltIn("ml-100k"), 1.514, 1.215)
}
func TestKNN(t *testing.T) {
	Evaluate(t, NewKNN(), LoadDataFromBuiltIn("ml-100k"), 0.98, 0.774)
}
func TestNMF(t *testing.T) {
	Evaluate(t, NewNMF(), LoadDataFromBuiltIn("ml-100k"), 0.963, 0.758)
}
func TestKNNWithMean(t *testing.T) {
	Evaluate(t, NewKNNWithMean(), LoadDataFromBuiltIn("ml-100k"), 0.951, 0.749)
}

func TestSlopeOne(t *testing.T) {
	Evaluate(t, NewSlopeOne(), LoadDataFromBuiltIn("ml-100k"), 0.946, 0.743)
}
func TestBaseLine(t *testing.T) {
	Evaluate(t, NewBaseLine(), LoadDataFromBuiltIn("ml-100k"), 0.944, 0.748)
}

func TestSVD(t *testing.T) {
	Evaluate(t, NewSVD(), LoadDataFromBuiltIn("ml-100k"), 0.934, 0.737)
}

func TestKNNBaseLine(t *testing.T) {
	Evaluate(t, NewKNNBaseLine(), LoadDataFromBuiltIn("ml-100k"), 0.931, 0.733)
}

// Comment out SVD++ test to avoid time out
func TestSVDPP(t *testing.T) {
	Evaluate(t, NewSVDPP(), LoadDataFromBuiltIn("ml-100k"), 0.92, 0.722)
}

func TestCoClustering(t *testing.T) {
	Evaluate(t, NewCoClustering(), LoadDataFromBuiltIn("ml-100k"), 0.963, 0.753)
}
