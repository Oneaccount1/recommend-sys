package main

import (
	"fmt"
	"github.com/olekukonko/tablewriter"
	"gonum.org/v1/gonum/stat"
	"os"
	"recommend-sys/core"
	"time"
)

func main() {

	dataset := "ml-100k"
	if len(os.Args) > 1 {
		dataset = os.Args[1]
	}

	// Cross validation
	estimators := map[string]core.Estimator{
		"Random":        core.NewRandom(nil),
		"Baseline":      core.NewBaseLine(nil),
		"SVD":           core.NewSVD(nil),
		"SVD++":         core.NewSVD(nil),
		"NMF":           core.NewNMF(nil),
		"Slope One":     core.NewSlopeOne(nil),
		"KNN":           core.NewKNN(nil),
		"Centered K-NN": core.NewKNNWithMean(nil),
		"K-NN Baseline": core.NewKNNBaseLine(nil),
		"K-NN Z-Score":  core.NewKNNWithZScore(nil),
		"Co-Clustering": core.NewCoClustering(nil),
	}
	set := core.LoadDataFromBuiltIn(dataset)

	// 基准测试输出
	var start time.Time
	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"Name", "RMSE", "MAE", "Time"})

	for name, algo := range estimators {
		start = time.Now()
		out := core.CrossValidate(algo, set, []core.Evaluator{core.RMSE, core.MAE},
			5, 0, nil)
		tm := time.Since(start)
		table.Append([]string{name,
			fmt.Sprintf("%.6f", stat.Mean(out[0].Tests, nil)),
			fmt.Sprintf("%.6f", stat.Mean(out[1].Tests, nil)),
			fmt.Sprint(tm),
		})
	}
	table.Render()
}
