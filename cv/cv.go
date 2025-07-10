package cv

import (
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/stat"
	"recommend-sys/core/algo"
	"recommend-sys/data"

	"math"
)

func AbsTo(dst, a []float64) {
	for i := 0; i < len(a); i++ {
		if a[i] < 0 {
			dst[i] = -a[i]
		} else {
			dst[i] = a[i]
		}
	}
}

func RootMeanSquareError(predictions, truth []float64) float64 {
	tmp := make([]float64, len(predictions))

	floats.SubTo(tmp, predictions, truth)
	floats.MulTo(tmp, tmp, tmp)

	return math.Sqrt(stat.Mean(tmp, nil))

}
func MeanAbsoluteError(predictions, truth []float64) float64 {
	// 平均值
	tmp := make([]float64, len(predictions))

	floats.SubTo(tmp, predictions, truth)

	AbsTo(tmp, tmp)

	return stat.Mean(tmp, nil)

}

// CrossValidate 验证推荐算法性能
func CrossValidate(recommender algo.Algorithm, dataSet data.Set, measure []string, cv int) [][]float64 {

	a := make([][]float64, cv)

	// 分割数据集
	trainSet, testSet := dataSet.KFold(cv)

	for i := 0; i < cv; i++ {
		trainFold := trainSet[i]
		testFold := testSet[i]

		// fit
		recommender.Fit(data.NewDataSet(trainFold))

		result := make([]float64, testFold.Nrow())
		// predict
		for j := 0; j < testFold.Nrow(); j++ {
			userID, _ := testFold.Elem(j, 0).Int()
			itemID, _ := testFold.Elem(j, 1).Int()

			predict := recommender.Predict(userID, itemID)
			result[j] = predict
		}
		tr := testFold.Col("X2").Float()
		a[i] = []float64{RootMeanSquareError(result, tr), MeanAbsoluteError(result, tr)}
	}
	return a
}
