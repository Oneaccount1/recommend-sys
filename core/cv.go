package core

// CrossValidate 验证推荐算法性能
func CrossValidate(algorithm Algorithm, dataSet TrainSet, metrics []Metrics, cv int, seed int64,
	options ...OptionSetter) [][]float64 {
	ret := make([][]float64, len(metrics))
	for i := 0; i < len(ret); i++ {
		ret[i] = make([]float64, cv)
	}
	// 分割测试集合
	trainFolds, testFolds := dataSet.KFold(cv, seed)

	for i := 0; i < cv; i++ {
		// 训练
		trainFold := trainFolds[i]
		testFold := testFolds[i]
		algorithm.Fit(trainFold, options...)
		// 获取test集相关数据
		interactionUsers, interactionItems, _ := testFold.Interactions()
		// 创建预测数组
		predictions := make([]float64, testFold.Length())

		for j := 0; j < testFold.Length(); j++ {
			userID := interactionUsers[j]
			itemID := interactionItems[j]
			// 预测
			predictions[j] = algorithm.Predict(userID, itemID)
		}
		_, _, truth := testFold.Interactions()

		// Metrics
		for j := 0; j < len(ret); j++ {
			ret[j][i] = metrics[j](predictions, truth)
		}

	}
	return ret
}
