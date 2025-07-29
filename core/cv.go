package core

import (
	"gonum.org/v1/gonum/stat"
	"math"
)

// ParameterGrid 实际上就是一个二维数组
type ParameterGrid map[string][]interface{}

type CrossValidateResult struct {
	Trains []float64
	Tests  []float64
}

// CrossValidate 验证推荐算法性能
func CrossValidate(algorithm Algorithm, dataSet DataSet, metrics []Evaluator, cv int, seed int64,
	params Parameters) []CrossValidateResult {

	ret := make([]CrossValidateResult, len(metrics))
	for i := 0; i < len(ret); i++ {
		ret[i].Trains = make([]float64, cv)
		ret[i].Tests = make([]float64, cv)
	}
	// 分割测试集合
	trainFolds, testFolds := dataSet.KFold(cv, seed)

	for i := 0; i < cv; i++ {
		// 训练
		trainFold := trainFolds[i]
		testFold := testFolds[i]
		algorithm.Fit(trainFold, params)
		// 获取test集相关数据
		interactionUsers, interactionItems := testFold.Users, testFold.Items
		// 创建预测数组
		predictions := make([]float64, testFold.Length())

		for j := 0; j < testFold.Length(); j++ {
			userID := interactionUsers[j]
			itemID := interactionItems[j]
			// 预测
			predictions[j] = algorithm.Predict(userID, itemID)
		}
		truth := testFold.Ratings

		// Metrics
		for j := 0; j < len(ret); j++ {
			ret[j].Tests[i] = metrics[j](predictions, truth)
		}

	}
	return ret
}

type GridSearchResult struct {
	BestScore float64
	BestParam map[string]interface{}
	BestIndex int
	CVResult  []CrossValidateResult
	AllParams []Parameters
}

// GridSearchCV Tune algorithm parameters with GridSearchCV
func GridSearchCV(algo Algorithm, dataSet DataSet, paramGrid ParameterGrid,
	evaluators []Evaluator, cv int, seed int64) []GridSearchResult {
	// 获取参数名字和长度
	params := make([]string, 0, len(paramGrid))
	count := 1
	for param, values := range paramGrid {
		params = append(params, param)
		// 计算每一种参数可能的组合
		count *= len(values)
	}
	// 创建网格搜索结果
	// 每种验证方式产生一种结果
	results := make([]GridSearchResult, len(evaluators))

	// 初始化
	for i := range results {
		results[i] = GridSearchResult{}
		results[i].BestScore = math.Inf(1)
		results[i].CVResult = make([]CrossValidateResult, 0, count)
		results[i].AllParams = make([]Parameters, 0, count)
	}

	// DFS
	var dfs func(deep int, options Parameters)

	dfs = func(deep int, options Parameters) {
		// 当deep == len(params) 时候，说明已经遍历完所有参数
		if deep == len(params) {
			// Cross validate
			cvResult := CrossValidate(algo, dataSet, evaluators, cv, seed, options)

			for i := range cvResult {
				results[i].CVResult = append(results[i].CVResult, cvResult[i])
				// 复制当前层的参数
				results[i].AllParams = append(results[i].AllParams, options.Copy())
				// 计算测试集平均分
				score := stat.Mean(cvResult[i].Tests, nil)
				if score < results[i].BestScore {
					results[i].BestScore = score
					results[i].BestParam = options.Copy()
					results[i].BestIndex = len(results[i].AllParams) - 1
				}
			}
		} else {
			// 选取下一个参数
			param := params[deep]
			values := paramGrid[param]
			// 遍历下一个参数的可能选值
			for _, val := range values {
				options[param] = val
				dfs(deep+1, options)
			}
		}
	}

	options := make(map[string]interface{})
	dfs(0, options)
	return results

}
